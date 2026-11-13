"""
scorenet.py - Génération 2 (v2) : le réseau prédit le SCORE FINAL ATTENDU d'une feuille.

  V(feuille) = base(feuille) + 20 * réseau(features)
  base = points déjà marqués + somme des valeurs moyennes des cases encore ouvertes
         (+ ~20 si la prime est encore atteignable)  -> le réseau n'apprend qu'une correction.

Deux modes de décision (NetPlayer-like) :
  "score" : maximise le score final attendu (jeu solitaire optimal approché)
  "win"   : maximise P(gagner) = Phi((V_moi - V_adversaire) / sigma), où sigma vient
            de l'incertitude restante (table 'sig' apprise : écart-type du score final
            selon le nombre de tours restants). Donne le comportement "1v1" :
            prudent quand on mène, risqué quand on est mené.
"""
import numpy as np

from yahtzee import N_DICE, N_CAT, YZ, T, KM, AVG, HeuristicPlayer, turn_tables
from netlib import MAXP, sheet_feat

SD = 34            # dimension des features d'une feuille
SCALE = 20.0       # unité de la correction du réseau (points)


def sheet_feat2(sh):
    f = np.empty(SD)
    f[:31] = sheet_feat(sh)
    up = sh.upper
    fu = sh.filled[:6]
    par = sum(3 * (c + 1) for c in range(6) if fu[c])
    open_max = sum(5 * (c + 1) for c in range(6) if not fu[c])
    f[31] = (up - par) / 15
    f[32] = (up + open_max - 63) / 63
    f[33] = (~sh.filled).sum() / 13
    return f


def base_value(sh):
    up = sh.upper
    op = ~sh.filled
    b = sh.pts.sum() + 35 * (up >= 63) + 100 * sh.yz_bonus + AVG[op].sum()
    if up < 63 and up + sum(5 * (c + 1) for c in range(6) if op[c]) >= 63:
        b += 20
    return b


def after_feats(me, C, Sv, B):
    """Features + base (vectorisées) des feuilles obtenues après chaque coup possible."""
    u = len(C)
    ar = np.arange(u)
    F = np.empty((u, SD))
    F[:, :13] = me.filled
    F[ar, C] = 1
    F[:, 13:26] = me.pts / MAXP
    F[ar, 13 + C] = Sv / MAXP[C]
    isup = C < 6
    up = me.upper + np.where(isup, Sv, 0.0)
    F[:, 26] = np.minimum(up, 63) / 63
    F[:, 27] = up >= 63
    F[:, 28] = (me.pts[YZ] == 50) | ((C == YZ) & (Sv == 50))
    nb = me.yz_bonus + B
    F[:, 29] = nb > 0
    tot = me.pts.sum() + Sv + 35 * (up >= 63) + 100 * nb
    F[:, 30] = tot / 300
    fu = np.tile(me.filled[:6], (u, 1))
    fu[ar[isup], C[isup]] = True
    faces = np.arange(1, 7)
    par = (fu * 3 * faces).sum(1)
    open_max = ((~fu) * 5 * faces).sum(1)
    F[:, 31] = (up - par) / 15
    F[:, 32] = (up + open_max - 63) / 63
    F[:, 33] = (13 - me.filled.sum() - 1) / 13
    op = np.tile(~me.filled, (u, 1))
    op[ar, C] = False
    base = tot + (op * AVG).sum(1) + 20 * ((up < 63) & (up + open_max >= 63))
    return F, base


class ScoreMLP:
    def __init__(self, h=128, seed=0, params=None):
        if params is not None:
            self.P = {k: np.array(v, dtype=float) for k, v in params.items()}
            return
        r = np.random.default_rng(seed)
        self.P = {
            "W1": r.standard_normal((SD, h)) * np.sqrt(2 / SD), "b1": np.zeros(h),
            "W2": r.standard_normal((h, h)) * np.sqrt(2 / h), "b2": np.zeros(h),
            "W3": r.standard_normal((h, 1)) * 0.01, "b3": np.zeros(1),
            "sig": 16.5 * np.sqrt(np.arange(14.0)),          # écart-type du score final / tours restants
        }

    def _fwd(self, X):
        P = self.P
        a1 = np.maximum(X @ P["W1"] + P["b1"], 0)
        a2 = np.maximum(a1 @ P["W2"] + P["b2"], 0)
        return (a2 @ P["W3"] + P["b3"])[:, 0], a1, a2

    def forward(self, X, base):
        return base + SCALE * self._fwd(X)[0]

    def grads(self, X, base, target):
        P = self.P
        out, a1, a2 = self._fwd(X)
        y = (target - base) / SCALE
        n = len(X)
        dz3 = (2 * (out - y) / n)[:, None]
        g = {"W3": a2.T @ dz3, "b3": dz3.sum(0)}
        dz2 = (dz3 @ P["W3"].T) * (a2 > 0)
        g["W2"], g["b2"] = a1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ P["W2"].T) * (a1 > 0)
        g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
        return g, float(np.mean((out - y) ** 2))

    def save(self, path):
        np.savez(path, **self.P)

    @staticmethod
    def load(path):
        d = np.load(path)
        return ScoreMLP(params={k: d[k] for k in d.files})


class ScorePlayer(HeuristicPlayer):
    def __init__(self, net, mode="win", name="scorenet"):
        super().__init__(name=name)
        self.net, self.mode, self._opp = net, mode, None

    def choose_keep(self, dice, rolls_left, me, opp):
        self._opp = opp
        return super().choose_keep(dice, rolls_left, me, opp)

    def choose_category(self, d, me, opp):
        self._opp = opp
        return super().choose_category(d, me, opp)

    def _prepare(self, me):
        opp = self._opp
        key = (me.filled.tobytes(), me.pts.tobytes(), me.yz_bonus)
        if self.mode == "win":
            key += (opp.filled.tobytes(), opp.pts.tobytes(), opp.yz_bonus)
        if key == self._key:
            return
        self._key = key
        S, legal, bonus = turn_tables(me)
        dd, cc = np.nonzero(legal)
        sv = S[dd, cc].astype(int)
        keys = (cc * 64 + sv) * 2 + (bonus[dd] > 0).astype(int)
        uniq, inv = np.unique(keys, return_inverse=True)
        C = (uniq >> 1) // 64
        Sv = ((uniq >> 1) % 64).astype(float)
        B = uniq & 1
        F, base = after_feats(me, C, Sv, B)
        V = self.net.forward(F, base)                      # score final attendu après chaque coup
        if self.mode == "win":
            vopp = self.net.forward(sheet_feat2(opp)[None], np.array([base_value(opp)]))[0]
            sig = self.net.P["sig"]
            n_me = 13 - int(me.filled.sum()) - 1
            n_opp = 13 - int(opp.filled.sum())
            sd = max(np.sqrt(sig[n_me] ** 2 + sig[n_opp] ** 2), 1.0)
            V = 1 / (1 + np.exp(-1.702 * (V - vopp) / sd))  # approx. de la loi normale
        val = np.full((N_DICE, N_CAT), -1e9)
        val[dd, cc] = V[np.ravel(inv)]
        W0 = val.max(axis=1)
        E1 = T @ W0
        W1 = E1[KM].max(axis=1)
        E2 = T @ W1
        self._val, self._W0, self._E = val, W0, {1: E1, 2: E2}
