"""
gen1_tune.py - Génération 1 : on règle les paramètres de HeuristicPlayer
par une stratégie d'évolution simple (échantillonner autour de la moyenne,
garder les meilleurs, recentrer, réduire le bruit).

Fitness = score moyen en solo sur des parties à graines communes (peu bruité et rapide).
Validation finale = match 1v1 contre le joueur de base (yahtzee.py).

Lancer :  python gen1_tune.py            (environ 5 à 10 min selon ton CPU)
Test    :  python gen1_tune.py --quick
Résultat:  best_params_gen1.json
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from yahtzee import HeuristicPlayer, DEFAULT_PARAMS, solo_average, arena

NAMES = list(DEFAULT_PARAMS)          # avg_w, k_upper, bonus_w


def to_params(x):
    return {n: float(max(v, 0.0)) for n, v in zip(NAMES, x)}


def fitness(args):
    x, seed, n_games = args
    return solo_average(HeuristicPlayer(to_params(x)), n_games=n_games, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--lam", type=int, default=16, help="candidats par génération")
    ap.add_argument("--games", type=int, default=500, help="parties solo par candidat")
    ap.add_argument("--final", type=int, default=4000, help="parties du match final 1v1")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.gens, a.lam, a.games, a.final = 3, 6, 100, 200

    rng = np.random.default_rng(0)
    mean = np.array([DEFAULT_PARAMS[n] for n in NAMES], dtype=float)
    sigma = 0.25
    mu = max(2, a.lam // 4)
    t0 = time.time()

    with Pool(os.cpu_count()) as pool:
        for g in range(a.gens):
            cands = [mean] + [mean * (1 + sigma * rng.standard_normal(len(mean)))
                              for _ in range(a.lam)]
            seed = 1000 + g                      # mêmes dés pour tous les candidats
            fits = pool.map(fitness, [(c, seed, a.games) for c in cands])
            order = np.argsort(fits)[::-1]
            elite = [cands[i] for i in order[:mu]]
            mean = np.mean(elite, axis=0)
            sigma *= 0.9
            print(f"gen {g+1:2d}/{a.gens}  meilleur={fits[order[0]]:.1f}  "
                  f"moyenne_actuelle={[round(float(v), 3) for v in mean]}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    best = to_params(mean)
    with open("best_params_gen1.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nParamètres retenus :", best)

    base = HeuristicPlayer(name="gen0")
    tuned = HeuristicPlayer(best, name="gen1")
    print(f"Solo (graine neuve) : gen0={solo_average(base, 1000, seed=777):.1f}  "
          f"gen1={solo_average(tuned, 1000, seed=777):.1f}")
    r = arena(tuned, base, a.final, seed=12345)
    print(f"Match 1v1 gen1 vs gen0 : {r['winrate']:.1%} ± {r['ci95']:.1%}  "
          f"(moy {r['avg_A']:.1f} vs {r['avg_B']:.1f})")


if __name__ == "__main__":
    main()
