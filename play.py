import requests
import json
import random
import os
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize')


agent_pool = {"plato": "http://127.0.0.1:8082/interact", "blender": "http://127.0.0.1:8080/interact", "dialoflow": "http://127.0.0.1:8089/interact", "dialogpt": "http://127.0.0.1:8086/interact"}

data = {}

start_utterance = "hello"


class PlatoAgent:
    def __init__(self, userid):
        self.userid = userid
        self.url = agent_pool["plato"]

    def act(self, text, replace=False):
        data = json.dumps({"userID": self.userid, "text": text, "replace": replace})
        r = requests.post(self.url, data=data)
        text = json.loads(r.text)['body']['utterance']
        return text

class DialoFlowAgent:
    def __init__(self, userid):
        self.userid = userid
        self.url = agent_pool["dialoflow"]

    def act(self, text, replace=False):
        data = json.dumps({"userID": self.userid, "text": text, "replace": replace})
        r = requests.post(self.url, data=data)
        text = json.loads(r.text)['body']['utterance']
        return text

class DialoGPTAgent:
    def __init__(self, userid):
        self.userid = userid
        self.url = agent_pool["dialogpt"]

    def act(self, text, replace=False):
        data = json.dumps({"userID": self.userid, "text": text, "replace": replace})
        r = requests.post(self.url, data=data)
        text = json.loads(r.text)['body']['utterance']
        return text



class BlenderAgent:
    def __init__(self, userid):
        self.userid = userid
        self.url = agent_pool["blender"]

    def act(self, text, replace=False):
        if replace:
            data = text+self.userid + "*"
        else:
            data = text+self.userid
        r = requests.post(self.url, data=data.encode("utf-8"))
        text = json.loads(r.text)["text"]
        return text

def gen_q(text):
    data = json.dumps({"text": text})
    url = "http://127.0.0.1:8084/gen"
    r = requests.post(url, data=data)
    text = json.loads(r.text)['body']['text']
    return text

def nli(res, res_gold):
    data = json.dumps({"res": res, "res_gold": res_gold})
    url = "http://127.0.0.1:8085/NLI"
    r = requests.post(url, data=data)
    score = json.loads(r.text)['nli_score']
    return score

PLAY_NUM = 1000
TURN = 15
METHOD = "GEN"
agent_name_pool = list(agent_pool.keys())
for i in range(PLAY_NUM):
    userid = random.randrange(100000, 999997)
    userid1 = str(userid+1)
    userid2 = str(userid+2)
    if userid in data.keys():
        continue
    else:
        data[userid] = []
    agent1_name = random.choice(agent_name_pool)
    if agent1_name == "plato":
        agent1 = PlatoAgent(userid1)
    elif agent1_name == "blender":
        agent1 = BlenderAgent(userid1)
    elif agent1_name == "dialogpt":
        agent1 = DialoGPTAgent(userid1)
    elif agent1_name == "dialoflow":
        agent1 = DialoFlowAgent(userid1)
    agent2_name = random.choice(agent_name_pool)
    if agent2_name == "plato":
        agent2 = PlatoAgent(userid2)
    elif agent2_name == "blender":
        agent2 = BlenderAgent(userid2)
    elif agent2_name == "dialogpt":
        agent2 = DialoGPTAgent(userid2)
    elif agent2_name == "dialoflow":
        agent2 = DialoFlowAgent(userid2)

    r1 = None
    r2 = None
    questions = []
    for j in range(TURN):
        if j == 0:
            r1 = start_utterance
            data[userid].append(r1+"\n")
        else:
            r1 = agent1.act(r2)
            print(r1)
            data[userid].append(r1+"\n")
        r2 = agent2.act(r1)
        print(r2)
        data[userid].append(r2+"\n")
        doc = nlp(r2)
        clean_r2 = []
        for k in doc.sentences:
            if "?" not in k.text:
                clean_r2.append(k.text)
        if len(clean_r2) == 0:
            continue
        q = gen_q(" ".join(clean_r2))
        print(" ".join(clean_r2), q)
        if len(q) > 0 and j > 0:
            q = random.choice(q)
            if len(q[0].strip()) > 0:
                data[userid].append("\t" + METHOD + ": " + q[0] + "\n")
                temp_r2 = agent2.act(q[0], replace=True)
                score = " ".join([str(x) for x in nli(temp_r2, q[1])])
                print(METHOD, temp_r2, score)
                data[userid].append("\t" + METHOD + ": " + temp_r2 + "\t" + q[1] + "\t" + score + "\n")

    if not os.path.exists(METHOD + "/" + agent1_name + "_" + agent2_name):
        os.mkdir(METHOD + "/" + agent1_name + "_" + agent2_name)
    with open(METHOD + "/" + agent1_name + "_" + agent2_name + '/' + str(userid), "w") as f:
        f.writelines(data[userid])







