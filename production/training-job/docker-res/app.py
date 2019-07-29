#!/usr/local/bin/python3
import logging, sys, os
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer
from utils import TransformerWithClfHead
from types import SimpleNamespace
from pydantic import BaseModel
from fastapi import FastAPI

logger = logging.getLogger("app.py")

LOG_DIR = os.getenv("LOG_PATH", "/logs")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Input(BaseModel):
    text: str


def load_files():
    # load metadata, parameters and tokenizer
    metadata = torch.load(LOG_DIR + "/metadata.bin")
    state_dict = torch.load(LOG_DIR + "/model_weights.pth",
                            map_location=device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=False)

    # create model
    model = TransformerWithClfHead(SimpleNamespace(**metadata["config"]),
                                   SimpleNamespace(**metadata["config_ft"]))

    model.load_state_dict(state_dict)
    return model, tokenizer, metadata


def predict(model, tokenizer, int2label, device=None, input="test"):
    "predict `input` with `model`"
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tok = tokenizer.tokenize(input)
    ids = tokenizer.convert_tokens_to_ids(tok) + [tokenizer.vocab['[CLS]']]
    tensor = torch.tensor(ids, dtype=torch.long)
    tensor = tensor.to(device)
    tensor = tensor.reshape(1, -1)
    tensor_in = tensor.transpose(0, 1).contiguous()  # [S, 1]
    logits = model(tensor_in,
                   clf_tokens_mask=(tensor_in == tokenizer.vocab['[CLS]']),
                   padding_mask=(tensor == tokenizer.vocab['[PAD]']))
    val, _ = torch.max(logits, 0)
    val = F.softmax(val, dim=0).detach().cpu().numpy()
    return {
        int2label[val.argmax()]: val.max(),
        int2label[val.argmin()]: val.min()
    }


app = FastAPI()

model, tokenizer, metadata = load_files()


@app.get("/")
def root():
    return {"message": "Not much here. Check out http://127.0.0.1:8000/docs!"}


@app.post("/inference/")
async def inference(input: Input):
    output = predict(model,
                     tokenizer,
                     metadata['int2label'],
                     device=device,
                     input=input.text)
    return {k: str(v) for k, v in output.items()}
