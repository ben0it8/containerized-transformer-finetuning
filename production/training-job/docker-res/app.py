#!/usr/local/bin/python3
from bottle import Bottle, run, request, response
import logging, sys
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
from json import dumps
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer
from utils import TransformerWithClfHead, FineTuningConfig, predict, getenv_cast

logger = logging.getLogger("app.py")

LOG_DIR = getenv_cast("LOG_PATH", cast=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load metadata, parameters and tokenizer
metadata = torch.load(LOG_DIR + "/metadata.bin")
state_dict = torch.load(LOG_DIR + "/model_weights.pth", map_location=device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=False)

# create model
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


logger.info(f"Inference web app for sentiment prediction started on device: {device}") 
logger.info("Go to http://0.0.0.0:5000/inference")
run(app, host="0.0.0.0", port=5000)