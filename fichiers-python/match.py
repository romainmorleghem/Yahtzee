"""
match.py - Fait s'affronter deux joueurs en 1v1 (parties alternées, intervalle de confiance).

Joueurs possibles :
  random        joueur aléatoire
  gen0          heuristique de base
  gen1          heuristique avec best_params_gen1.json
  fichier.npz   réseau de neurones. Pour les réseaux "score" (scorenet_*.npz) on peut
                choisir le mode : scorenet_best.npz:win (défaut, maximise P(gagner))
                                  scorenet_best.npz:score (maximise le score attendu)

Exemples :
  python match.py gen1 gen0 2000
  python match.py gen2_best.npz gen1 2000
"""
import json
import os
import sys
from multiprocessing import Pool

from yahtzee import HeuristicPlayer, RandomPlayer, DEFAULT_PARAMS, arena
from netlib import MLP, NetPlayer
from scorenet import ScoreMLP, ScorePlayer


def load_player(spec):
    if spec == "random":
        return RandomPlayer(0)
    if spec == "gen0":
        return HeuristicPlayer(name="gen0")
    if spec == "gen1":
        p = dict(DEFAULT_PARAMS)
        p.update(json.load(open("best_params_gen1.json")))
        return HeuristicPlayer(p, name="gen1")
    path, _, mode = spec.partition(":")
    import numpy as np
    if "sig" in np.load(path).files:
        return ScorePlayer(ScoreMLP.load(path), mode=mode or "win", name=spec)
    return NetPlayer(MLP.load(path), name=spec)


def chunk(args):
    sa, sb, n, seed = args
    r = arena(load_player(sa), load_player(sb), n, seed)
    return r["winrate"] * n, r["avg_A"] * n, r["avg_B"] * n, n


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    sa, sb, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    w = os.cpu_count() or 1
    per = max(2, n // w)
    with Pool(w) as pool:
        res = pool.map(chunk, [(sa, sb, per, 5000 + i) for i in range(w)])
    N = sum(r[3] for r in res)
    p = sum(r[0] for r in res) / N
    ci = 1.96 * (max(p * (1 - p), 1e-9) / N) ** 0.5
    print(f"{sa} vs {sb} sur {N} parties : {p:.1%} ± {ci:.1%}  "
          f"(moy {sum(r[1] for r in res)/N:.1f} vs {sum(r[2] for r in res)/N:.1f})")
