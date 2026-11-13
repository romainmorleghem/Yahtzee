"""Convertit scorenet_best.npz en weights.js (lu par la page web).
Usage : python export_weights.py scorenet_best.npz     (numpy requis)"""
import sys, json
import numpy as np
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "scorenet_best.npz")
r = lambda a: np.round(np.asarray(a, float), 5).tolist()
net = {"W1": r(d["W1"]), "b1": r(d["b1"]), "W2": r(d["W2"]), "b2": r(d["b2"]),
       "W3": r(d["W3"][:, 0]), "b3": float(d["b3"][0]), "sig": r(d["sig"])}
open("weights.js", "w").write("window.YAHTZEE_NET=" + json.dumps(net, separators=(",", ":")) + ";")
print("weights.js écrit")
