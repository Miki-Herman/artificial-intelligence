import json
import time
import os

def rozdej_skore(tah1, tah2):
    # 1 = zradi, 0 = nezradi

    skores = (0, 0)

    if (tah1 == 1) and (tah2 == 1):
        skores = (2, 2)

    if (tah1 == 1) and (tah2 == 0):
        skores = (0, 3)

    if (tah1 == 0) and (tah2 == 1):
        skores = (3, 0)

    if (tah1 == 0) and (tah2 == 0):
        skores = (1, 1)

    return skores


def play(f1, f2, stepsnum):

    skore1 = 0
    skore2 = 0

    historie1 = []
    historie2 = []

    for i in range(stepsnum):
        tah1 = f1(historie1, historie2)
        tah2 = f2(historie2, historie1)

        s1, s2 = rozdej_skore(tah1, tah2)
        skore1 += s1
        skore2 += s2

        historie1.append(tah1)
        historie2.append(tah2)

    return skore1, skore2

def write_best_genom(genom, participants, score):

    data = {
        "created_at": f"{int(time.time())}",
        "genom": genom,
        "trained_on": [i.__name__ for i in participants],
        "score": score
    }

    if os.path.getsize("best_genom.json") > 0 and os.path.exists("best_genom.json"):
        with open("best_genom.json", "r") as f:
            loaded_data = json.load(f)

        with open("genom_history.json", "r") as f:
            history_data = json.load(f)

        if score < loaded_data["score"]:
            with open("genom_history.json", "w") as f:
                history_data["history"].append(loaded_data)
                f.write(json.dumps(history_data, indent=4, sort_keys=True, default=str))

            with open("best_genom.json", "w") as f:
                f.write(json.dumps(data, indent=4, sort_keys=True, default=str))

        if score > loaded_data["score"]:
            with open("genom_history.json", "w") as f:
                history_data["history"].append(data)
                f.write(json.dumps(history_data, indent=4, sort_keys=True, default=str))

    else:
        with open("best_genom.json", "w") as f:
            f.write(json.dumps(data, indent=4, sort_keys=True, default=str))

def load_best_genom():
    with open("best_genom.json", "r") as f:
        data = json.load(f)
        return data