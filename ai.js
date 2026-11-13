// ai.js - moteur Yahtzee + réseau + décisions (port fidèle de yahtzee.py / scorenet.py)
(function () {
'use strict';
const NC = 13, YZ = 11;
const AVG = [2.11,5.28,8.57,12.16,15.69,19.19,21.66,13.10,22.59,29.46,32.71,16.87,22.01];
const MAXP = [5,10,15,20,25,30,30,30,25,30,40,50,30];
const NAMES = ["As","Deux","Trois","Quatre","Cinq","Six","Brelan","Carré","Full","Petite suite","Grande suite","Yahtzee","Chance"];
const key = a => [...a].sort((x, y) => x - y).join('');

const DICE = [];
(function g(p, m) { if (p.length === 5) { DICE.push(p.slice()); return; } for (let v = m; v <= 6; v++) { p.push(v); g(p, v); p.pop(); } })([], 1);
const DIDX = new Map(DICE.map((d, i) => [d.join(''), i]));

function raw(d, c) {
  const cnt = [0,0,0,0,0,0]; d.forEach(v => cnt[v - 1]++);
  const s = d.reduce((a, b) => a + b, 0), mx = Math.max(...cnt), has = a => cnt[a - 1] > 0;
  if (c < 6) return (c + 1) * cnt[c];
  if (c === 6) return mx >= 3 ? s : 0;
  if (c === 7) return mx >= 4 ? s : 0;
  if (c === 8) return cnt.includes(3) && cnt.includes(2) ? 25 : 0;
  if (c === 9) return [1,2,3].some(a => has(a) && has(a+1) && has(a+2) && has(a+3)) ? 30 : 0;
  if (c === 10) return [1,2].some(a => [0,1,2,3,4].every(k => has(a + k))) ? 40 : 0;
  if (c === 11) return mx === 5 ? 50 : 0;
  return s;
}
const SC = DICE.map(d => Array.from({ length: NC }, (_, c) => raw(d, c)));

const KEEPS = [], KIDX = new Map();
(function g(p, m) { KIDX.set(p.join(''), KEEPS.length); KEEPS.push(p.slice()); if (p.length === 5) return; for (let v = m; v <= 6; v++) { p.push(v); g(p, v); p.pop(); } })([], 1);
const T = new Float64Array(KEEPS.length * 252);
KEEPS.forEach((K, ki) => {
  const n = 5 - K.length, tot = Math.pow(6, n);
  (function r(o, k) { if (k === n) { T[ki * 252 + DIDX.get(key(K.concat(o)))] += 1 / tot; return; } for (let v = 1; v <= 6; v++) { o.push(v); r(o, k + 1); o.pop(); } })([], 0);
});
const KM = new Int32Array(252 * 32);
DICE.forEach((d, i) => { for (let m = 0; m < 32; m++) KM[i * 32 + m] = KIDX.get(key(d.filter((_, j) => m >> j & 1))); });
const mv = W => { const o = new Float64Array(KEEPS.length); for (let k = 0; k < o.length; k++) { let s = 0; for (let j = 0; j < 252; j++) s += T[k * 252 + j] * W[j]; o[k] = s; } return o; };

// ---- feuille de score
const newSheet = () => ({ f: Array(NC).fill(false), p: Array(NC).fill(0), yb: 0 });
const upper = s => s.p.slice(0, 6).reduce((a, b) => a + b, 0);
const total = s => s.p.reduce((a, b) => a + b, 0) + (upper(s) >= 63 ? 35 : 0) + 100 * s.yb;

function tables(sh) {   // points, coups légaux, bonus pour chacune des 252 mains (règle du Joker incluse)
  const S = SC.map(r => r.slice()), L = DICE.map(() => sh.f.map(x => !x)), B = new Array(252).fill(0);
  if (sh.f[YZ]) DICE.forEach((d, i) => {
    if (d[0] !== d[4]) return;
    const face = d[0] - 1; S[i][8] = 25; S[i][9] = 30; S[i][10] = 40;
    if (sh.p[YZ] === 50) B[i] = 100;
    if (!sh.f[face]) L[i] = L[i].map((_, c) => c === face);
    else if ([6,7,8,9,10,12].some(c => !sh.f[c])) for (let c = 0; c < 6; c++) L[i][c] = false;
  });
  return { S, L, B };
}
function apply(sh, d, c) {
  const t = tables(sh), i = DIDX.get(key(d));
  sh.p[c] = t.S[i][c]; sh.f[c] = true; if (t.B[i]) sh.yb++;
}

// ---- réseau : valeur = score final attendu
function sfeat(s) {
  const up = upper(s), f = [];
  for (let c = 0; c < NC; c++) f.push(s.f[c] ? 1 : 0);
  for (let c = 0; c < NC; c++) f.push(s.p[c] / MAXP[c]);
  f.push(Math.min(up, 63) / 63, up >= 63 ? 1 : 0, s.p[YZ] === 50 ? 1 : 0, s.yb > 0 ? 1 : 0, total(s) / 300);
  let par = 0, om = 0;
  for (let c = 0; c < 6; c++) { if (s.f[c]) par += 3 * (c + 1); else om += 5 * (c + 1); }
  f.push((up - par) / 15, (up + om - 63) / 63, s.f.filter(x => !x).length / 13);
  return f;
}
function baseV(s) {
  const up = upper(s); let b = total(s), om = 0;
  for (let c = 0; c < NC; c++) if (!s.f[c]) b += AVG[c];
  for (let c = 0; c < 6; c++) if (!s.f[c]) om += 5 * (c + 1);
  if (up < 63 && up + om >= 63) b += 20;
  return b;
}
function nn(x) {
  const N = window.YAHTZEE_NET, h = N.b1.length, a1 = Float64Array.from(N.b1), a2 = Float64Array.from(N.b2);
  for (let i = 0; i < x.length; i++) { const xi = x[i]; if (xi) { const w = N.W1[i]; for (let j = 0; j < h; j++) a1[j] += xi * w[j]; } }
  for (let j = 0; j < h; j++) if (a1[j] < 0) a1[j] = 0;
  for (let i = 0; i < h; i++) { const v = a1[i]; if (v) { const w = N.W2[i]; for (let j = 0; j < h; j++) a2[j] += v * w[j]; } }
  let o = N.b3;
  for (let j = 0; j < h; j++) if (a2[j] > 0) o += a2[j] * N.W3[j];
  return o;
}
const value = s => baseV(s) + 20 * nn(sfeat(s));

class AI {
  constructor(mode) { this.mode = mode; }            // 'score' ou 'win'
  plan(me, opp) {
    const { S, L, B } = tables(me), cache = new Map(), val = DICE.map(() => Array(NC).fill(-1e9));
    const win = this.mode === 'win', sig = window.YAHTZEE_NET.sig;
    let vopp = 0, sd = 1;
    if (win) {
      const nMe = 13 - (me.f.filter(Boolean).length + 1), nOpp = 13 - opp.f.filter(Boolean).length;
      vopp = value(opp); sd = Math.max(Math.hypot(sig[nMe], sig[nOpp]), 1);
    }
    for (let i = 0; i < 252; i++) for (let c = 0; c < NC; c++) {
      if (!L[i][c]) continue;
      const b = B[i] > 0 ? 1 : 0, s = S[i][c], k = (c * 64 + s) * 2 + b;
      let v = cache.get(k);
      if (v === undefined) {
        const t = { f: me.f.slice(), p: me.p.slice(), yb: me.yb + b }; t.f[c] = true; t.p[c] = s;
        v = value(t); if (win) v = 1 / (1 + Math.exp(-1.702 * (v - vopp) / sd));
        cache.set(k, v);
      }
      val[i][c] = v;
    }
    const W0 = val.map(r => Math.max(...r)), E1 = mv(W0);
    const W1 = DICE.map((_, i) => { let m = -1e18; for (let k = 0; k < 32; k++) { const e = E1[KM[i * 32 + k]]; if (e > m) m = e; } return m; });
    this.val = val; this.W0 = W0; this.E = { 1: E1, 2: mv(W1) };
  }
  chooseKeep(dice, r) {   // renvoie 5 booléens (true = garder)
    const E = this.E[r], stop = this.W0[DIDX.get(key(dice))];
    let best = -1e18, bm = 0;
    for (let m = 0; m < 31; m++) { const v = E[KIDX.get(key(dice.filter((_, i) => m >> i & 1)))]; if (v > best) { best = v; bm = m; } }
    return stop >= best - 1e-9 ? dice.map(() => true) : dice.map((_, i) => !!(bm >> i & 1));
  }
  chooseCategory(d) {
    const v = this.val[DIDX.get(key(d))]; let bc = 0;
    for (let c = 1; c < NC; c++) if (v[c] > v[bc]) bc = c;
    return bc;
  }
}
const API = { AI, newSheet, tables, apply, total, upper, NAMES, key, DIDX, value, sfeat };
if (typeof module !== 'undefined') module.exports = API; else window.YAI = API;
})();
