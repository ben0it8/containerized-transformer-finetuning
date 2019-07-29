import sys
from subprocess import call
from utils import *

logger = logging.getLogger("run.py")

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# path to data, logging, resources
DATA_DIR = getenv_cast("DATA_PATH", cast=str)
LOG_DIR = getenv_cast("LOG_PATH", cast=str)
RESOURCES_DIR = getenv_cast("RESOURCES_PATH", cast=str)

# path to IMDB
IMDB_DIR = os.path.join(DATA_DIR, "imdb5k")

# url to IMDB data, pretraining args and checkpoint
IMDB_URL = "https://github.com/ben0it8/transformer-finetuning/raw/master/imdb5k.tar.gz"
PRETRAINING_ARGS_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/model_training_args.bin"
PRETRAINING_CKPT_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/model_checkpoint.pth"

# get configs from env vars
MAX_TRAIN = getenv_cast("NUM_TRAIN", cast=int)
MAX_TEST = getenv_cast("NUM_TEST", cast=int)
NUM_MAX_POSITIONS = getenv_cast("NUM_MAX_POSITIONS", cast=int)
if NUM_MAX_POSITIONS > 256:
    logger.warning(f"`NUM_MAX_POSITIONS` has to be smaller than 256")
    NUM_MAX_POSITIONS = 256
VALID_PCT = getenv_cast("VALID_PCT", cast=float)
N_EPOCHS = getenv_cast("N_EPOCHS", cast=int)
BATCH_SIZE = getenv_cast("BATCH_SIZE", cast=int)
OMP_NUM_THREADS = getenv_cast("BATCH_SIZE", cast=int)
SEED = getenv_cast("SEED", cast=int)
LR = getenv_cast("LR", cast=float)
MAX_NORM = getenv_cast("MAX_NORM", cast=float)
TRAIN_BODY = getenv_cast("TRAIN_BODY", cast=int)

# set number of threads for this process and random seed
torch.set_num_threads(OMP_NUM_THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)

# make code device agnostic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define rest of parameters
FineTuningConfig = namedtuple(
    'FineTuningConfig',
    field_names=
    "num_classes, dropout, init_range, batch_size, lr, max_norm, n_epochs,"
    "n_warmup, valid_pct, gradient_acc_steps, device, log_dir")

finetuning_config = FineTuningConfig(2, 0.05, 0.02, BATCH_SIZE, LR, MAX_NORM,
                                     N_EPOCHS, 10, VALID_PCT, 2, device,
                                     LOG_DIR)

if __name__ == "__main__":

    t0 = time()
    # download imdb dataset
    file_path = download_url(IMDB_URL, '/tmp', overwrite=True)

    # untar imdb dataset to DATA_DIR
    untar(file_path, DATA_DIR)

    # read data, 5000-5000 each
    datasets = read_imdb(IMDB_DIR,
                         max_lengths={
                             "train": MAX_TRAIN,
                             "test": MAX_TEST
                         })

    # list of labels
    labels = list(set(datasets["train"][LABEL_COL].tolist()))

    # labels to integers mapping
    label2int = {label: i for i, label in enumerate(labels)}

    # download the 'bert-base-cased' tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)

    # initialize a TextProcessor
    processor = TextProcessor(tokenizer,
                              label2int,
                              num_max_positions=NUM_MAX_POSITIONS)

    # create train and valid sets by splitting
    train_dl, valid_dl = create_dataloader(
        datasets["train"],
        processor,
        batch_size=finetuning_config.batch_size,
        valid_pct=finetuning_config.valid_pct)
    test_dl = create_dataloader(datasets["test"],
                                processor,
                                batch_size=finetuning_config.batch_size,
                                valid_pct=None)

    # download pre-training config and weights
    config = torch.load(cached_path(PRETRAINING_ARGS_URL))
    state_dict = torch.load(cached_path(PRETRAINING_CKPT_URL),
                            map_location='cpu')

    # init model: Transformer base + classifier head
    model = TransformerWithClfHead(config=config,
                                   fine_tuning_config=finetuning_config).to(
                                       finetuning_config.device)
    model.load_state_dict(state_dict, strict=False)

    if not TRAIN_BODY:
        logger.warning(
            f"TRAIN_BODY is set to {TRAIN_BODY}, only training the classifier head!"
        )
        freeze_body(model)

    optimizer = AdamW(model.parameters(), lr=finetuning_config.lr)

    def update(engine, batch):
        "update function for training"
        model.train()
        inputs, labels = (t.to(finetuning_config.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # [S, B]
        _, loss = model(
            inputs,
            clf_tokens_mask=(inputs == tokenizer.vocab[processor.CLS]),
            clf_labels=labels)
        loss = loss / finetuning_config.gradient_acc_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       finetuning_config.max_norm)
        if engine.state.iteration % finetuning_config.gradient_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def inference(engine, batch):
        "update function for evaluation"
        model.eval()
        with torch.no_grad():
            batch, labels = (t.to(finetuning_config.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()
            logits = model(
                inputs,
                clf_tokens_mask=(inputs == tokenizer.vocab[processor.CLS]),
                padding_mask=(batch == tokenizer.vocab[processor.PAD]))
        return logits, labels

    trainer = Engine(update)
    evaluator = Engine(inference)

    # add metric to evaluator
    Accuracy().attach(evaluator, "accuracy")

    # add evaluator to trainer: eval on valid set after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_dl)
        logger.info(
            f"Validation epoch {engine.state.epoch},  accuracy: {100*evaluator.state.metrics['accuracy']}"
        )

    # lr schedule: linearly warm-up to lr and then to zero
    scheduler = PiecewiseLinear(
        optimizer, 'lr', [(0, 0.0),
                          (finetuning_config.n_warmup, finetuning_config.lr),
                          (len(train_dl) * finetuning_config.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # save checkpoints and finetuning config
    checkpoint_handler = ModelCheckpoint(finetuning_config.log_dir,
                                         'finetuning_checkpoint',
                                         save_interval=1,
                                         require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
                              {'imdb_model': model})

    int2label = {i: label for label, i in label2int.items()}

    # save metadata
    torch.save(
        {
            "config": config.__dict__,
            "config_ft": finetuning_config._asdict(),
            "int2label": int2label
        }, os.path.join(finetuning_config.log_dir, "metadata.bin"))

    @timeit
    def train():
        "fit the model on `train_dl`"
        logger.info(finetuning_config)
        trainer.run(train_dl, max_epochs=finetuning_config.n_epochs)
        # save model weights
        torch.save(
            model.state_dict(),
            os.path.join(finetuning_config.log_dir, "model_weights.pth"))

    @timeit
    def eval():
        "evaluate the model on `test_dl`"
        evaluator.run(test_dl)
        logger.info(
            f"Test accuracy: {100*evaluator.state.metrics['accuracy']:.3f}")

    try:
        logger.warning(f"Training started on device: {device}")
        train()
    except Exception as error:
        logger.error(error)
        raise

    eval()

    training_time = timedelta(seconds=round(time() - t0, 1))
    logger.info(f"Training finished in {str(training_time):0>8}")

    call(f"cd {RESOURCES_DIR} && uvicorn app:app --host 0.0.0.0 --port 8000", shell=True)
    sys.exit(0)
