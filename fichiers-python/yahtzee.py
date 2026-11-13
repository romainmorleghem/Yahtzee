"""
yahtzee.py - Moteur Yahtzee 1v1, joueur heuristique "humain expérimenté", arène.

Dépendance : numpy uniquement.
Lancer : python yahtzee.py

Architecture (pensée pour brancher plus tard un réseau de neurones) :
  - Sheet            : feuille de score d'un joueur
  - turn_tables()    : scores / coups légaux / bonus pour chaque main possible (252 mains)
  - Player           : interface = choose_keep(...) et choose_category(...)
  - HeuristicPlayer  : stratégie de base (prime, valeur moyenne des cases...)
  - arena()          : matchs 1v1 avec intervalle de confiance
"""
import itertools
import math
import random
import numpy as np

# --------------------------------------------------------------------------
# 1. Constantes et tables précalculées
# --------------------------------------------------------------------------
CATS = ["As", "Deux", "Trois", "Quatre", "Cinq", "Six",
        "Brelan", "Carré", "Full", "Petite suite", "Grande suite",
        "Yahtzee", "Chance"]
N_CAT = 13
YZ = 11                      # index de la case Yahtzee

# 252 mains possibles (5 dés triés)
DICE = list(itertools.combinations_with_replacement(range(1, 7), 5))
DICE_IDX = {d: i for i, d in enumerate(DICE)}
N_DICE = len(DICE)


def _raw_score(d, c):
    cnt = [d.count(f) for f in range(1, 7)]
    s = sum(d)
    faces = set(d)
    if c < 6:
        return (c + 1) * cnt[c]
    if c == 6:
        return s if max(cnt) >= 3 else 0
    if c == 7:
        return s if max(cnt) >= 4 else 0
    if c == 8:
        return 25 if sorted(cnt)[-2:] == [2, 3] else 0
    if c == 9:
        return 30 if any({a, a + 1, a + 2, a + 3} <= faces for a in (1, 2, 3)) else 0
    if c == 10:
        return 40 if faces in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if c == 11:
        return 50 if max(cnt) == 5 else 0
    return s


SCORES = np.array([[_raw_score(d, c) for c in range(N_CAT)] for d in DICE], dtype=float)
YZ_ROWS = [i for i, d in enumerate(DICE) if d[0] == d[4]]

# Sous-ensembles de dés "gardés" (multi-ensembles de taille 0..5) -> 462
KEEPS = sorted({tuple(sorted(c)) for n in range(6)
                for c in itertools.combinations_with_replacement(range(1, 7), n)},
               key=lambda k: (len(k), k))
KEEP_IDX = {k: i for i, k in enumerate(KEEPS)}

# T[k, d] = proba d'obtenir la main d en gardant k et relançant le reste
T = np.zeros((len(KEEPS), N_DICE))
for ki, K in enumerate(KEEPS):
    n = 5 - len(K)
    for out in itertools.product(range(1, 7), repeat=n):
        T[ki, DICE_IDX[tuple(sorted(K + out))]] += 1.0 / 6 ** n

# KM[d, m] = index du multi-ensemble gardé si on garde les positions du masque m
KM = np.zeros((N_DICE, 32), dtype=int)
for i, d in enumerate(DICE):
    for m in range(32):
        KM[i, m] = KEEP_IDX[tuple(sorted(d[j] for j in range(5) if m >> j & 1))]

# Valeur moyenne de chaque case sous jeu optimal en solitaire (Verhoeff)
AVG = np.array([2.11, 5.28, 8.57, 12.16, 15.69, 19.19,
                21.66, 13.10, 22.59, 29.46, 32.71, 16.87, 22.01])


# --------------------------------------------------------------------------
# 2. Feuille de score + règles (avec Joker et bonus Yahtzee)
# --------------------------------------------------------------------------
class Sheet:
    def __init__(self):
        self.filled = np.zeros(N_CAT, dtype=bool)
        self.pts = np.zeros(N_CAT)
        self.yz_bonus = 0

    @property
    def upper(self):
        return self.pts[:6].sum()

    def total(self):
        return self.pts.sum() + (35 if self.upper >= 63 else 0) + 100 * self.yz_bonus

    def done(self):
        return self.filled.all()


def turn_tables(sh):
    """Pour chacune des 252 mains : points, coups légaux, bonus Yahtzee (+100)."""
    open_ = ~sh.filled
    S = SCORES.copy()
    legal = np.tile(open_, (N_DICE, 1))
    bonus = np.zeros(N_DICE)
    if sh.filled[YZ]:                       # règle du Joker
        for i in YZ_ROWS:
            face = DICE[i][0] - 1
            S[i, 8], S[i, 9], S[i, 10] = 25, 30, 40
            if sh.pts[YZ] == 50:
                bonus[i] = 100
            if open_[face]:                 # case haute correspondante obligatoire
                legal[i, :] = False
                legal[i, face] = True
            elif open_[6:11].any() or open_[12]:   # sinon une case basse
                legal[i, :6] = False
    return S, legal, bonus


def apply_score(sh, d, cat):
    S, legal, bonus = turn_tables(sh)
    i = DICE_IDX[d]
    assert legal[i, cat], "coup illégal"
    sh.pts[cat] = S[i, cat]
    sh.filled[cat] = True
    if bonus[i]:
        sh.yz_bonus += 1


# --------------------------------------------------------------------------
# 3. Joueurs
# --------------------------------------------------------------------------
class RandomPlayer:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_keep(self, dice, rolls_left, me, opp):
        return [self.rng.random() < 0.5 for _ in range(5)]

    def choose_category(self, d, me, opp):
        _, legal, _ = turn_tables(me)
        return self.rng.choice(list(np.flatnonzero(legal[DICE_IDX[d]])))


DEFAULT_PARAMS = dict(
    avg_w=1.0,        # poids du "coût d'opportunité" (valeur moyenne de la case)
    k_upper=0.6,      # valeur d'un point au-dessus/au-dessous de la "moyenne" (3 par face)
    bonus_w=1.0,      # poids du bonus de 35 quand on le franchit
)


class HeuristicPlayer:
    """
    Joueur "humain qui connaît les bases". Il évalue chaque fin de tour par :
        points - valeur moyenne de la case (on ne gâche pas une case précieuse)
        + gestion de la prime (63 pts en haut -> +35)
        + bonus Yahtzee
    puis calcule exactement, par programmation dynamique sur la main (252 mains),
    quels dés garder pour maximiser cette valeur sur les relances restantes.
    """
    name = "heuristic"

    def __init__(self, params=None, name=None):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        if name:
            self.name = name
        self._key = None

    def _prepare(self, me):
        key = (me.filled.tobytes(), int(me.upper), me.pts[YZ] == 50)
        if key == self._key:
            return
        self._key = key
        S, legal, bonus = turn_tables(me)
        val = S + bonus[:, None] - self.p["avg_w"] * AVG[None, :]

        up = me.upper
        open_up = ~me.filled[:6]
        reachable = up < 63 and up + sum(5 * (c + 1) for c in range(6) if open_up[c]) >= 63
        for c in range(6):
            after = up + S[:, c]
            cross = (up < 63) & (after >= 63)
            term = self.p["k_upper"] * (S[:, c] - 3 * (c + 1)) if reachable else 0.0
            val[:, c] += np.where(cross, 35 * self.p["bonus_w"], term)

        val = np.where(legal, val, -1e9)
        W0 = val.max(axis=1)              # fin de tour
        E1 = T @ W0                       # valeur de chaque "garde" avec 1 relance
        W1 = E1[KM].max(axis=1)
        E2 = T @ W1                       # valeur de chaque "garde" avec 2 relances
        self._val, self._W0, self._E = val, W0, {1: E1, 2: E2}

    def choose_keep(self, dice, rolls_left, me, opp):
        self._prepare(me)
        E = self._E[rolls_left]
        stop = self._W0[DICE_IDX[tuple(sorted(dice))]]
        best, bm = -1e18, 0
        for m in range(31):
            K = tuple(sorted(dice[i] for i in range(5) if m >> i & 1))
            v = E[KEEP_IDX[K]]
            if v > best:
                best, bm = v, m
        if stop >= best - 1e-9:
            return [True] * 5
        return [bool(bm >> i & 1) for i in range(5)]

    def choose_category(self, d, me, opp):
        self._prepare(me)
        return int(np.argmax(self._val[DICE_IDX[d]]))


# --------------------------------------------------------------------------
# 4. Moteur de partie et arène
# --------------------------------------------------------------------------
def play_turn(me, opp, player, rng):
    dice = [rng.randint(1, 6) for _ in range(5)]
    for rolls_left in (2, 1):
        keep = player.choose_keep(tuple(dice), rolls_left, me, opp)
        if all(keep):
            break
        dice = [dv if k else rng.randint(1, 6) for dv, k in zip(dice, keep)]
    d = tuple(sorted(dice))
    cat = player.choose_category(d, me, opp)
    apply_score(me, d, cat)
    return d, cat


def play_game(pA, pB, rng, verbose=False):
    """A joue en premier. Retourne (total A, total B)."""
    a, b = Sheet(), Sheet()
    for t in range(13):
        for who, sh, other, pl in (("A", a, b, pA), ("B", b, a, pB)):
            d, cat = play_turn(sh, other, pl, rng)
            if verbose:
                print(f"tour {t+1:2d} {who} {d} -> {CATS[cat]:<12} (A={a.total():.0f}  B={b.total():.0f})")
    return a.total(), b.total()


def arena(pA, pB, n_games=1000, seed=0):
    """n parties, A et B alternent le rôle de premier joueur. Retourne dict de stats."""
    rng = random.Random(seed)
    wins = 0.0
    sa = sb = 0.0
    for g in range(n_games):
        if g % 2 == 0:
            x, y = play_game(pA, pB, rng)
        else:
            y, x = play_game(pB, pA, rng)
        sa += x
        sb += y
        wins += 1.0 if x > y else 0.5 if x == y else 0.0
    p = wins / n_games
    ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n_games)
    return dict(winrate=p, ci95=ci, avg_A=sa / n_games, avg_B=sb / n_games)


def solo_average(player, n_games=300, seed=1):
    """Score moyen en solitaire (repère : optimal = ~254.6)."""
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(n_games):
        sh, dummy = Sheet(), Sheet()
        for _ in range(13):
            play_turn(sh, dummy, player, rng)
        tot += sh.total()
    return tot / n_games


if __name__ == "__main__":
    import time
    h = HeuristicPlayer()
    t0 = time.time()
    print("Partie exemple :")
    play_game(h, RandomPlayer(1), random.Random(42), verbose=True)
    print(f"\nScore moyen solo heuristique : {solo_average(h):.1f}  (optimal ~254.6)")
    r = arena(h, RandomPlayer(2), 500)
    print(f"Heuristique vs random : {r['winrate']:.1%} ± {r['ci95']:.1%}  "
          f"(moy {r['avg_A']:.0f} vs {r['avg_B']:.0f})")
    print(f"({time.time()-t0:.1f}s)")
