import json
import torch
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformers import *
app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("MNLI/")
model = AutoModelForSequenceClassification.from_pretrained("MNLI/")
model.to("cuda")
model.eval()
sm = torch.nn.Softmax()

def score(c1, c2):
    c1_t = torch.LongTensor(tokenizer(c1, padding="longest")["input_ids"]).cuda()
    c2_t = torch.LongTensor(tokenizer(c2, padding="longest")["input_ids"]).cuda()
    text = torch.cat([c2_t, c1_t], dim=-1).unsqueeze(0)
    return sm(model(text)[0])

@app.route("/NLI", methods=["POST"])
def nli():
    body = request.get_data()
    body = json.loads(body)
    res = body["res"]
    res_gold = body["res_gold"]
    results = score(res, res_gold).cpu().tolist()[0]

    return jsonify({"nli_score": results})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8085)
