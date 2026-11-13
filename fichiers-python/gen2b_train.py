"""
gen2b_train.py - Génération 2 (v2) : apprentissage du score final attendu (TD(lambda), solitaire).

Principe : l'adversaire n'influence pas la valeur "score attendu" d'une feuille, donc on
s'entraîne en parties SOLITAIRES (2x plus rapide, signal beaucoup plus riche qu'un 0/1).
Le réseau apprend une correction par rapport à la base (points + valeurs moyennes des cases).
La table 'sig' (incertitude selon le nombre de tours restants) est estimée au passage et
sert ensuite au mode "win" (1v1).

Lancer :  python gen2b_train.py --quick
          python gen2b_train.py                      (vrai entraînement)
          python gen2b_train.py --iters 300          (plus long)
Reprendre: python gen2b_train.py --resume scorenet_latest.npz
Sorties  : scorenet_latest.npz, scorenet_best.npz (meilleur score solo), scorenet_it<N>.npz
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from yahtzee import (Sheet, HeuristicPlayer, DEFAULT_PARAMS, DICE_IDX, apply_score,
                     turn_tables, solo_average, arena)
from netlib import Adam
from scorenet import ScoreMLP, ScorePlayer, sheet_feat2, base_value, SD


def solo_worker(args):
    params, n_games, seed, eps = args
    rng = random.Random(seed)
    pl = ScorePlayer(ScoreMLP(params=params), mode="score")
    dummy = Sheet()
    Fs, Bs, fin = [], [], []
    for _ in range(n_games):
        sh = Sheet()
        fr, br = [], []
        for _t in range(13):
            dice = [rng.randint(1, 6) for _ in range(5)]
            for r in (2, 1):
                keep = pl.choose_keep(tuple(dice), r, sh, dummy)
                if all(keep):
                    break
                dice = [dv if k else rng.randint(1, 6) for dv, k in zip(dice, keep)]
            d = tuple(sorted(dice))
            cat = pl.choose_category(d, sh, dummy)
            if eps and rng.random() < eps:
                _, legal, _ = turn_tables(sh)
                cat = rng.choice(list(np.flatnonzero(legal[DICE_IDX[d]])))
            apply_score(sh, d, cat)
            fr.append(sheet_feat2(sh))
            br.append(base_value(sh))
        Fs.append(fr); Bs.append(br); fin.append(sh.total())
    return np.array(Fs), np.array(Bs), np.array(fin)


def eval_solo_worker(args):
    params, n, seed = args
    return solo_average(ScorePlayer(ScoreMLP(params=params), mode="score"), n, seed) * n


def eval_win_worker(args):
    params, heur_params, n, seed = args
    r = arena(ScorePlayer(ScoreMLP(params=params), mode="win"), HeuristicPlayer(heur_params), n, seed)
    return r["winrate"], r["avg_A"], r["avg_B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--games", type=int, default=200, help="parties solo par itération")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-4, help="taux d'apprentissage final (décroissance en cosinus)")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--eps-final", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=600)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.iters, a.games, a.eval_every, a.eval_games = 6, 40, 3, 60

    heur_params = dict(DEFAULT_PARAMS)
    if os.path.exists("best_params_gen1.json"):
        heur_params.update(json.load(open("best_params_gen1.json")))

    net = ScoreMLP.load(a.resume) if a.resume else ScoreMLP(h=a.hidden)
    opt = Adam(net, a.lr)
    nw = os.cpu_count() or 1
    buffer, best, t0 = [], -1.0, time.time()

    with Pool(nw) as pool:
        for it in range(1, a.iters + 1):
            frac = 0.5 * (1 + np.cos(np.pi * (it - 1) / max(a.iters - 1, 1)))   # 1 -> 0
            opt.lr = a.lr_final + (a.lr - a.lr_final) * frac
            eps = a.eps_final + (a.eps - a.eps_final) * frac
            snap = {k: v.copy() for k, v in net.P.items()}
            per = max(1, a.games // nw)
            res = pool.map(solo_worker, [(snap, per, 100000 * it + w, eps) for w in range(nw)])
            F = np.concatenate([r[0] for r in res])
            B = np.concatenate([r[1] for r in res])
            fin = np.concatenate([r[2] for r in res])
            buffer.append((F, B, fin))
            buffer = buffer[-a.buffer:]

            F = np.concatenate([b[0] for b in buffer])
            B = np.concatenate([b[1] for b in buffer])
            fin = np.concatenate([b[2] for b in buffer])
            n = len(F)
            V = net.forward(F.reshape(-1, SD), B.reshape(-1)).reshape(n, 13)
            G = np.empty((n, 13))
            G[:, -1] = fin
            for k in range(11, -1, -1):                   # cibles TD(lambda)
                G[:, k] = (1 - a.lam) * V[:, k + 1] + a.lam * G[:, k + 1]

            # table d'incertitude : écart-type de (score final - V) selon les tours restants
            resid = fin[:, None] - V                      # colonne k = après k+1 coups -> 12-k restants
            sig_new = np.array([max(resid[:, 12 - r].std(), 1.0) if r <= 12 else 0 for r in range(14)])
            sig_new[0] = 1.0
            sig_new[13] = sig_new[12] * np.sqrt(13 / 12)
            net.P["sig"] = 0.8 * net.P["sig"] + 0.2 * sig_new

            X, Bf, y = F.reshape(-1, SD), B.reshape(-1), G.reshape(-1)
            losses = []
            for _ in range(a.epochs):
                perm = np.random.permutation(len(X))
                for i in range(0, len(X), a.batch):
                    idx = perm[i:i + a.batch]
                    g, loss = net.grads(X[idx], Bf[idx], y[idx])
                    opt.step(g)
                    losses.append(loss)

            line = (f"it {it:3d}/{a.iters}  score parties d'entraînement={res[0][2].mean():.1f}  "
                    f"loss={np.mean(losses):.3f}  ({time.time()-t0:.0f}s)")
            net.save("scorenet_latest.npz")
            if it % a.eval_every == 0:
                snap = {k: v.copy() for k, v in net.P.items()}
                per = max(2, a.eval_games // nw)
                solo = sum(pool.map(eval_solo_worker, [(snap, per, 999 + w) for w in range(nw)])) / (per * nw)
                ev = pool.map(eval_win_worker, [(snap, heur_params, per, 777 + w) for w in range(nw)])
                wr = float(np.mean([e[0] for e in ev]))
                line += f"  | solo={solo:.1f}  1v1 'win' vs heuristique={wr:.1%}"
                net.save(f"scorenet_it{it}.npz")
                if solo > best:
                    best = solo
                    net.save("scorenet_best.npz")
            print(line, flush=True)
    print(f"Terminé. Meilleur score solo mesuré : {best:.1f}")


if __name__ == "__main__":
    main()
