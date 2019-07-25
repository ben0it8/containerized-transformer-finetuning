from bottle import Bottle, run, request, response
from json import dumps

import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer
from utils import TransformerWithClfHead, FineTuningConfig, predict

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
    print(f"text: {text}")
    output = predict(model,
                     tokenizer,
                     metadata['int2label'],
                     device=device,
                     input=text)
    print(output)
    response.content_type = "application/json"
    return dumps({k: str(v) for k, v in output.items()}, indent=4)


run(app, host="0.0.0.0", port=5000, debug=True)