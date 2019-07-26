#!/usr/local/bin/python3
"""app.py: toy webapp predicting the senntiment of input text at `localhost:5000/inference`"""

__author__ = "Oliver Atanaszov"
__email__ = "oliver.atanaszov@gmail.com"
__github__ = "https://github.com/ben0it8"
__copyright__ = "Copyright 2019, Planet Earth"

from bottle import Bottle, run, request, response
import logging
logging.basicConfig(level=logging.WARNING)
from json import dumps
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer
from utils import TransformerWithClfHead, FineTuningConfig, predict

logger = logging.getLogger("app.py")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=False)

metadata = torch.load("/logs/metadata.bin")
state_dict = torch.load("/logs/model_weights.pth", map_location=device)

model = TransformerWithClfHead(metadata["config"], metadata["config_ft"])
model.load_state_dict(state_dict)

app = Bottle()


@app.route("/inference")
def inference():
    return '''
        <form action="/inference" method="post">
            Text: <input name="text" type="text" />
            <input value="Inference" type="submit" />
        </form>
        '''


@app.route("/inference", method='POST')
def do_inference():
    text = request.forms.get("text")
    print(f"input: {text}")
    output = predict(model,
                     tokenizer,
                     metadata['int2label'],
                     device=device,
                     input=text)
    print(f"prediction: {output}")
    response.content_type = "application/json"
    return dumps({k: str(v) for k, v in output.items()}, indent=4)


print(f"\nWeb app for movie sentiment predictio started on device: {device} ")
run(app, host="0.0.0.0", port=5000)