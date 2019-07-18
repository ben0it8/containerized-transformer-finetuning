"""
TODO
* multiprocessing for dataloader
"""

import sys
from utils import *

import logging
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("run.py")

# path to data 
DATA_DIR = getenv_cast("DATA_PATH", cast=str)

# path to IMDB
IMDB_DIR = os.path.join(DATA_DIR, "imdb5k")

# url to IMDB data, pretraining args and checkpoint
IMDB_URL = "https://github.com/ben0it8/transformer-finetuning/raw/master/imdb5k.tar.gz"
PRETRAINING_ARGS_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/model_training_args.bin"
PRETRAINING_CKPT_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/model_training_args.bin"

# get configs from env vars
MAX_TRAIN = getenv_cast("NUM_TRAIN_SAMPLES", cast=int)
MAX_TEST = getenv_cast("NUM_TEST_SAMPLES", cast=int)
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

# set number of threads for this process and random seed
torch.set_num_threads(OMP_NUM_THREADS)
torch.manual_seed(SEED)
np.random.seed(SEED)

LOG_DIR = "./logs/"
CACHE_DIR = "./cache/"

# make code device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

FineTuningConfig = namedtuple(
    'FineTuningConfig',
    field_names=
    "num_classes, dropout, init_range, batch_size, lr, max_norm, n_epochs,"
    "n_warmup, valid_pct, gradient_acc_steps, device, log_dir, dataset_cache")

finetuning_config = FineTuningConfig(
    2, 0.1, 0.02, BATCH_SIZE, LR, MAX_NORM, N_EPOCHS,
    10, VALID_PCT, 4, device, LOG_DIR, 
    CACHE_DIR+'dataset_cache.bin')


if __name__ == "__main__":
    
    t0 = time()
    # download imdb dataset
    file_path = download_url(IMDB_URL, '/tmp', overwrite=True)

    # untar imdb dataset to DATA_DIR
    untar(file_path, DATA_DIR)

    # read data, 5000-5000 each
    datasets = read_imdb(IMDB_DIR, max_lengths={"train": MAX_TRAIN, "test": MAX_TEST})

    # list of labels
    labels = list(set(datasets["train"][LABEL_COL].tolist()))

    # labels to integers mapping
    label2int = {label: i for i, label in enumerate(labels)}

    # download the 'bert-base-cased' tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # initialize a TextProcessor
    processor = TextProcessor(tokenizer, label2int, num_max_positions=NUM_MAX_POSITIONS)

    # create train and valid sets by splitting
    train_dl, valid_dl = create_dataloader(datasets["train"], processor, 
                                            batch_size=finetuning_config.batch_size, 
                                            valid_pct=finetuning_config.valid_pct)
    test_dl = create_dataloader(datasets["test"], processor, 
                                 batch_size=finetuning_config.batch_size, 
                                 valid_pct=None)
    # download pre-trained model and config
    
    state_dict = torch.load(cached_path(PRETRAINING_CKPT_URL), map_location='cpu')
    config = torch.load(cached_path(PRETRAINING_ARGS_URL))

    # init model: Transformer base + classifier head
    model = TransformerWithClfHead(config=config, fine_tuning_config=finetuning_config).to(finetuning_config.device)
    optimizer = AdamW(model.parameters(), lr=finetuning_config.lr)

    def update(engine, batch):
        "update function for training"
        model.train()
        inputs, labels = (t.to(finetuning_config.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous() # [S, B]
        _, loss = model(inputs, 
                        clf_tokens_mask = (inputs == tokenizer.vocab[processor.CLS]), 
                        clf_labels=labels)
        loss = loss / finetuning_config.gradient_acc_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), finetuning_config.max_norm)
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
            logits = model(inputs,
                           clf_tokens_mask = (inputs == tokenizer.vocab[processor.CLS]),
                           padding_mask = (batch == tokenizer.vocab[processor.PAD]))
        return logits, labels

    trainer = Engine(update)
    evaluator = Engine(inference)
    
    # add metric to evaluator 
    Accuracy().attach(evaluator, "accuracy")

    # add evaluator to trainer: eval on valid set after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_dl)
        print(f"Validation epoch {engine.state.epoch},  accuracy: {100*evaluator.state.metrics['accuracy']}")

    # lr schedule: linearly warm-up to lr and then to zero
    scheduler = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (finetuning_config.n_warmup, finetuning_config.lr),
                                    (len(train_dl)*finetuning_config.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # save checkpoints and finetuning config
    checkpoint_handler = ModelCheckpoint(finetuning_config.log_dir, 'finetuning_checkpoint', 
                                         save_interval=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'imdb_model': model})

    # save config to logdir
    torch.save(finetuning_config, os.path.join(finetuning_config.log_dir, 'fine_tuning_args.bin'))          
    
    @timeit
    def train():
        # fit the model on train_dl
        trainer.run(train_dl, max_epochs=finetuning_config.n_epochs)

    @timeit
    def eval():
        # evaluate the model on test_dl
        evaluator.run(test_dl)
        print(f"Test accuracy: {100*evaluator.state.metrics['accuracy']:.3f}")

    train()
    eval()

    job_time = timedelta(seconds = round(time()-t0,1))
    print(f"Job finished, runtime: {str(job_time):0>8} ")
    sys.exit(0)
