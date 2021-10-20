import json
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import stanza
import truecase
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,pos')

app = Flask(__name__)
CORS(app)

from text2text.text_generator import TextGenerator
qg = TextGenerator(output_type="question")

def extract(features, select_list=None):
    upos = ""
    xpos = ""
    results_upos = []
    results_xpos = []
    temp_upos_text = []
    temp_xpos_text = []
    for feature in features.to_dict():
        for i in feature:
            if upos != i["upos"]:
                results_upos.append((" ".join(temp_upos_text), upos))
                temp_upos_text = [i["text"]]
                upos = i["upos"]
            else:
                temp_upos_text.append(i["text"])
            if xpos != i["xpos"]:
                results_xpos.append((" ".join(temp_xpos_text), xpos))
                temp_xpos_text = [i["text"]]
                xpos = i["xpos"]
            else:
                temp_xpos_text.append(i["text"])

    if select_list is not None:
        r_list = [i.to_dict()["text"] for i in features.entities]
        for i in results_xpos:
            if i[1] in select_list:
                if i[0] not in " ".join(r_list):
                    r_list.append(i[0])
        return r_list
    else:
        return results_xpos

@app.route("/gen", methods=["POST"])
def gen():
    body = request.get_data()
    body = json.loads(body)
    text = truecase.get_true_case(body["text"])
    select_list = ["CD", "NNP"]
    doc = nlp(text)
    ents = extract(doc, select_list)
    #ents = [ent.text for sent in doc.sentences for ent in sent.ents]
    print(ents)
    candidates = [text + " [SEP] " + i for i in ents]
    q = qg.predict(candidates)
    results = []
    for i in q:
        if i != "\n":
            results.append((i[0].replace(" I ", " you ").replace(" i ", " you ").replace(" my ", " your "), text))
    return jsonify({"body": {"text": results}})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8084)
