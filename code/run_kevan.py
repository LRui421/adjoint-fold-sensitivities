import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp2 as X
t = [t for t in X.build_tasks() if t["label"] == "EMP_kevan1970"][0]
t = dict(t); t["sub"] = False
res = X.analyze(t)
res["note"] = "run without greedy/local-search/misspecification (computational cost)"
pickle.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2", "EMP_kevan1970.pkl"), "wb"))
print("done", res.get("runtime_s"))
