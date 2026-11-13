"""
gen2_train.py - Génération 2 : réseau de neurones V(état) = P(je gagne), appris par
TD(lambda) en auto-jeu.

Phases :
  1) "warm" : le réseau apprend d'abord en regardant l'heuristique gen1 jouer contre
     elle-même (départ raisonnable, sinon un réseau aléatoire joue n'importe comment).
  2) "self-play" : le réseau joue (glouton sur sa propre valeur) contre lui-même (50 %)
     ou contre un pool d'adversaires : heuristique gen1 + anciennes versions de lui-même.

Sorties :  gen2_latest.npz, gen2_best.npz (meilleur vs heuristique), gen2_it<N>.npz

Lancer :  python gen2_train.py --quick          (test, ~1-2 min)
          python gen2_train.py                  (vrai entraînement, long)
          python gen2_train.py --torch          (apprentissage avec torch au lieu de numpy)
Reprendre : python gen2_train.py --resume gen2_latest.npz --warm 0
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from yahtzee import (Sheet, HeuristicPlayer, DEFAULT_PARAMS, DICE_IDX, apply_score,
                     turn_tables, arena)
from netlib import MLP, Adam, NetPlayer, state_feat


# ------------------------------------------------------------ auto-jeu (workers)
def make_player(spec):
    kind, payload = spec
    if kind == "heur":
        return HeuristicPlayer(payload)
    return NetPlayer(MLP(params=payload))


def turn_record(me, opp, player, rng, eps):
    dice = [rng.randint(1, 6) for _ in range(5)]
    for r in (2, 1):
        keep = player.choose_keep(tuple(dice), r, me, opp)
        if all(keep):
            break
        dice = [dv if k else rng.randint(1, 6) for dv, k in zip(dice, keep)]
    d = tuple(sorted(dice))
    cat = player.choose_category(d, me, opp)
    if eps and rng.random() < eps:                       # exploration
        _, legal, _ = turn_tables(me)
        cat = rng.choice(list(np.flatnonzero(legal[DICE_IDX[d]])))
    apply_score(me, d, cat)


def selfplay_worker(args):
    learner_spec, pool_specs, n_games, seed, eps, p_self = args
    rng = random.Random(seed)
    learner = make_player(learner_spec)
    pool = [make_player(s) for s in pool_specs]
    seqs, outs = [], []
    for _ in range(n_games):
        self_play = (not pool) or rng.random() < p_self
        opp = learner if self_play else rng.choice(pool)
        learner_first = rng.random() < 0.5
        a, b = Sheet(), Sheet()                          # a = apprenant, b = adversaire
        sa, sb = [], []
        for _t in range(13):
            for who in (("L", "O") if learner_first else ("O", "L")):
                if who == "L":
                    turn_record(a, b, learner, rng, eps)
                    sa.append(state_feat(a, b))
                else:
                    turn_record(b, a, opp, rng, eps)
                    if self_play:
                        sb.append(state_feat(b, a))
        ta, tb = a.total(), b.total()
        res = 1.0 if ta > tb else 0.5 if ta == tb else 0.0
        seqs.append(np.array(sa)); outs.append(res)
        if self_play:
            seqs.append(np.array(sb)); outs.append(1.0 - res)
    return np.array(seqs), np.array(outs)


def eval_worker(args):
    net_params, heur_params, n, seed = args
    r = arena(NetPlayer(MLP(params=net_params)), HeuristicPlayer(heur_params), n, seed)
    return r["winrate"], r["avg_A"], r["avg_B"]


# ------------------------------------------------------------ apprentissage
def lambda_returns(net, seqs, outs, lam):
    n, L, D = seqs.shape
    p = net.forward(seqs.reshape(-1, D)).reshape(n, L)
    G = np.empty((n, L))
    G[:, -1] = outs
    for k in range(L - 2, -1, -1):
        G[:, k] = (1 - lam) * p[:, k + 1] + lam * G[:, k + 1]
    return G


class NumpyTrainer:
    def __init__(self, net, lr):
        self.net, self.opt = net, Adam(net, lr)

    def step(self, X, t):
        g, loss = self.net.grads(X, t)
        self.opt.step(g)
        return loss

    def sync(self):
        pass


class TorchTrainer:
    """Même apprentissage mais avec l'autograd de torch (poids recopiés dans net.P)."""

    def __init__(self, net, lr):
        import torch                                     # importé tard, après la création du Pool
        self.torch, self.net = torch, net
        self.p = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
                  for k, v in net.P.items()}
        self.opt = torch.optim.Adam(self.p.values(), lr=lr)

    def step(self, X, t):
        torch, p = self.torch, self.p
        X, t = torch.from_numpy(X), torch.from_numpy(t)
        a1 = torch.relu(X @ p["W1"] + p["b1"])
        a2 = torch.relu(a1 @ p["W2"] + p["b2"])
        z = (a2 @ p["W3"] + p["b3"])[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def sync(self):
        for k, v in self.p.items():
            self.net.P[k] = v.detach().numpy().copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warm", type=int, default=6, help="itérations d'échauffement (heuristique)")
    ap.add_argument("--games", type=int, default=200, help="parties par itération")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=4, help="nb d'itérations gardées en mémoire")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.03, help="proba de case au hasard (exploration)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.iters, a.warm, a.games, a.eval_every, a.eval_games, a.snap_every = 4, 2, 40, 2, 40, 2

    heur_params = dict(DEFAULT_PARAMS)
    if os.path.exists("best_params_gen1.json"):
        heur_params.update(json.load(open("best_params_gen1.json")))
        print("Heuristique gen1 chargée :", heur_params)
    else:
        print("best_params_gen1.json absent -> heuristique de base")

    net = MLP.load(a.resume) if a.resume else MLP(h=a.hidden, seed=0)
    n_workers = os.cpu_count() or 1
    pool_nets = []                                       # anciennes versions du réseau
    buffer = []
    best_wr, t0 = -1.0, time.time()

    with Pool(n_workers) as pool:                        # créé AVANT d'importer torch
        trainer = (TorchTrainer if a.torch else NumpyTrainer)(net, a.lr)
        for it in range(1, a.iters + 1):
            warm = it <= a.warm
            if warm:
                learner = ("heur", heur_params)
                opp_pool, p_self = [], 1.0
            else:
                learner = ("net", {k: v.copy() for k, v in net.P.items()})
                opp_pool = [("heur", heur_params)] + [("net", p) for p in pool_nets[-5:]]
                p_self = 0.5
            per = max(1, a.games // n_workers)
            jobs = [(learner, opp_pool, per, 100000 * it + w, a.eps, p_self)
                    for w in range(n_workers)]
            res = pool.map(selfplay_worker, jobs)
            seqs = np.concatenate([r[0] for r in res])
            outs = np.concatenate([r[1] for r in res])
            buffer.append((seqs, outs))
            buffer = buffer[-a.buffer:]

            S = np.concatenate([b[0] for b in buffer])
            O = np.concatenate([b[1] for b in buffer])
            G = lambda_returns(net, S, O, a.lam)         # cibles recalculées avec le réseau actuel
            X = S.reshape(-1, S.shape[-1])
            y = G.reshape(-1)
            losses = []
            for _ in range(a.epochs):
                perm = np.random.permutation(len(X))
                for i in range(0, len(X), a.batch):
                    idx = perm[i:i + a.batch]
                    losses.append(trainer.step(X[idx], y[idx]))
            trainer.sync()

            line = (f"it {it:3d}/{a.iters} {'warm' if warm else 'self'}  "
                    f"parties={len(outs)}  loss={np.mean(losses):.4f}  ({time.time()-t0:.0f}s)")
            net.save("gen2_latest.npz")
            if it % a.snap_every == 0 and not warm:
                pool_nets.append({k: v.copy() for k, v in net.P.items()})
            if it % a.eval_every == 0:
                n_each = max(2, a.eval_games // n_workers)
                ev = pool.map(eval_worker, [({k: v.copy() for k, v in net.P.items()},
                                             heur_params, n_each, 777 + w)
                                            for w in range(n_workers)])
                wr = float(np.mean([e[0] for e in ev]))
                ma = float(np.mean([e[1] for e in ev]))
                mb = float(np.mean([e[2] for e in ev]))
                line += f"  | vs heuristique: {wr:.1%} (moy {ma:.0f} vs {mb:.0f})"
                net.save(f"gen2_it{it}.npz")
                if wr > best_wr:
                    best_wr = wr
                    net.save("gen2_best.npz")
            print(line, flush=True)
    print("Terminé. Meilleur taux de victoire mesuré :", f"{best_wr:.1%}")


if __name__ == "__main__":
    main()
