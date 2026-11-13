"""
netlib.py - MLP en numpy pur (pas de torch), features d'état, et NetPlayer.

Le réseau estime V(état) = probabilité que "moi" gagne la partie, à partir de
l'état APRÈS mon coup (feuille de score des deux joueurs). NetPlayer réutilise
exactement la même mécanique de relances que HeuristicPlayer, mais la valeur
d'une fin de tour est donnée par le réseau au lieu de "points - moyenne de la case".
"""
import numpy as np

from yahtzee import (DICE_IDX, N_DICE, N_CAT, YZ, T, KM, HeuristicPlayer, turn_tables)

# points max par case (pour normaliser)
MAXP = np.array([5, 10, 15, 20, 25, 30, 30, 30, 25, 30, 40, 50, 30], dtype=float)

SHEET_DIM = 31
STATE_DIM = 2 * SHEET_DIM + 3          # 65


# ---------------------------------------------------------------- features
def sheet_feat(sh):
    up = sh.upper
    f = np.empty(SHEET_DIM)
    f[:13] = sh.filled
    f[13:26] = sh.pts / MAXP
    f[26] = min(up, 63) / 63
    f[27] = up >= 63
    f[28] = sh.pts[YZ] == 50
    f[29] = sh.yz_bonus > 0
    f[30] = sh.total() / 300
    return f


def state_feat(me, opp):
    return np.concatenate([
        sheet_feat(me), sheet_feat(opp),
        [(me.total() - opp.total()) / 100, me.filled.sum() / 13, opp.filled.sum() / 13]])


def afterstate_feats(me, opp, C, Sv, B):
    """Features (vectorisées) de plusieurs états "après coup" possibles.
    C = case jouée, Sv = points marqués, B = 1 si bonus Yahtzee (+100)."""
    u = len(C)
    ar = np.arange(u)
    F = np.empty((u, STATE_DIM))
    F[:, :13] = me.filled
    F[ar, C] = 1
    F[:, 13:26] = me.pts / MAXP
    F[ar, 13 + C] = Sv / MAXP[C]
    up = me.upper + np.where(C < 6, Sv, 0.0)
    F[:, 26] = np.minimum(up, 63) / 63
    F[:, 27] = up >= 63
    F[:, 28] = (me.pts[YZ] == 50) | ((C == YZ) & (Sv == 50))
    nb = me.yz_bonus + B
    F[:, 29] = nb > 0
    tot = me.pts.sum() + Sv + 35 * (up >= 63) + 100 * nb
    F[:, 30] = tot / 300
    F[:, 31:62] = sheet_feat(opp)
    F[:, 62] = (tot - opp.total()) / 100
    F[:, 63] = (me.filled.sum() + 1) / 13
    F[:, 64] = opp.filled.sum() / 13
    return F


# ---------------------------------------------------------------- réseau
class MLP:
    def __init__(self, d_in=STATE_DIM, h=64, seed=0, params=None):
        if params is not None:
            self.P = {k: np.array(v, dtype=float) for k, v in params.items()}
            return
        r = np.random.default_rng(seed)
        self.P = {
            "W1": r.standard_normal((d_in, h)) * np.sqrt(2 / d_in), "b1": np.zeros(h),
            "W2": r.standard_normal((h, h)) * np.sqrt(2 / h), "b2": np.zeros(h),
            "W3": r.standard_normal((h, 1)) * np.sqrt(1 / h), "b3": np.zeros(1),
        }

    def forward(self, X, cache=False):
        P = self.P
        a1 = np.maximum(X @ P["W1"] + P["b1"], 0)
        a2 = np.maximum(a1 @ P["W2"] + P["b2"], 0)
        z = (a2 @ P["W3"] + P["b3"])[:, 0]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return (p, (X, a1, a2)) if cache else p

    def grads(self, X, t):
        """Gradient de l'entropie croisée (cibles t dans [0,1])."""
        P = self.P
        p, (X, a1, a2) = self.forward(X, cache=True)
        n = len(X)
        dz3 = ((p - t) / n)[:, None]
        g = {"W3": a2.T @ dz3, "b3": dz3.sum(0)}
        dz2 = (dz3 @ P["W3"].T) * (a2 > 0)
        g["W2"], g["b2"] = a1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ P["W2"].T) * (a1 > 0)
        g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
        loss = -np.mean(t * np.log(p + 1e-9) + (1 - t) * np.log(1 - p + 1e-9))
        return g, loss

    def save(self, path):
        np.savez(path, **self.P)

    @staticmethod
    def load(path):
        d = np.load(path)
        return MLP(params={k: d[k] for k in d.files})


class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999):
        self.net, self.lr, self.b1, self.b2 = net, lr, b1, b2
        self.m = {k: np.zeros_like(v) for k, v in net.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.P.items()}
        self.t = 0

    def step(self, g):
        self.t += 1
        for k in g:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.net.P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------- joueur
class NetPlayer(HeuristicPlayer):
    """Même mécanique de relances que HeuristicPlayer, valeur de fin de tour = réseau."""

    def __init__(self, net, name="net"):
        super().__init__(name=name)
        self.net = net
        self._opp = None

    def choose_keep(self, dice, rolls_left, me, opp):
        self._opp = opp
        return super().choose_keep(dice, rolls_left, me, opp)

    def choose_category(self, d, me, opp):
        self._opp = opp
        return super().choose_category(d, me, opp)

    def _prepare(self, me):
        opp = self._opp
        key = (me.filled.tobytes(), me.pts.tobytes(), me.yz_bonus,
               opp.filled.tobytes(), opp.pts.tobytes(), opp.yz_bonus)
        if key == self._key:
            return
        self._key = key
        S, legal, bonus = turn_tables(me)
        dd, cc = np.nonzero(legal)
        sv = S[dd, cc].astype(int)
        keys = (cc * 64 + sv) * 2 + (bonus[dd] > 0).astype(int)
        uniq, inv = np.unique(keys, return_inverse=True)       # états "après coup" distincts
        C = (uniq >> 1) // 64
        Sv = ((uniq >> 1) % 64).astype(float)
        B = uniq & 1
        V = self.net.forward(afterstate_feats(me, opp, C, Sv, B))
        val = np.full((N_DICE, N_CAT), -1e9)
        val[dd, cc] = V[np.ravel(inv)]
        W0 = val.max(axis=1)
        E1 = T @ W0
        W1 = E1[KM].max(axis=1)
        E2 = T @ W1
        self._val, self._W0, self._E = val, W0, {1: E1, 2: E2}
