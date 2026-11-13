// app.js - interface de jeu
(() => {
const { AI, newSheet, tables, apply, total, upper, NAMES, key, DIDX } = window.YAI;
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms * G.spd));
const PIPS = { 1:[4], 2:[0,8], 3:[0,4,8], 4:[0,2,6,8], 5:[0,2,4,6,8], 6:[0,2,3,5,6,8] };
const IDX5 = [0,1,2,3,4];
let G = { spd: 1 };

// ---- construction de l'affichage
const diceEl = IDX5.map((i) => { const d = document.createElement('div'); d.className = 'die';
  d.innerHTML = '<i></i>'.repeat(9); d.onclick = () => toggle(i); $('dice').appendChild(d); return d; });
const ROWS = [...[0,1,2,3,4,5].map(c => ({ c })), { k:'sub' }, { k:'bon' }, ...[6,7,8,9,10,11,12].map(c => ({ c })), { k:'yb' }, { k:'tot' }];
function buildCard() {
  $('card').innerHTML = `<tr><th></th><th class="you">Toi</th><th class="ai">IA ${G.mode === 'win' ? 'Win' : 'Score'}</th></tr>` +
    ROWS.map((r, i) => `<tr class="${r.k === 'tot' ? 'tot' : r.k ? 'sum' : ''}"><td>${r.c !== undefined ? NAMES[r.c] :
      { sub: 'Sous-total haut (/63)', bon: 'Prime +35', yb: 'Bonus Yahtzee', tot: 'TOTAL' }[r.k]}</td><td></td><td></td></tr>`).join('');
}

function render() {
  const me = G.sh[0], rolled = G.rolls < 3, myTurn = G.turn === 0 && !G.busy && !G.over;
  diceEl.forEach((el, i) => {
    const v = G.dice[i]; el.className = 'die' + (v ? '' : ' empty') + (G.held[i] ? ' held' : '') +
      (G.rolling.includes(i) ? ' rolling' : '') + (myTurn && G.rolls > 0 && rolled ? ' can' : '');
    [...el.children].forEach((p, k) => p.className = v && PIPS[v].includes(k) ? 'on' : '');
  });
  $('roll').disabled = !(myTurn && G.rolls > 0);
  $('roll').textContent = G.rolls === 3 ? 'Lancer les dés' : `Relancer (${G.rolls} restant${G.rolls > 1 ? 's' : ''})`;
  const rows = $('card').rows, t = rolled && G.dice.every(Boolean) ? tables(me) : null;
  const di = t ? DIDX.get(key(G.dice)) : -1;
  ROWS.forEach((r, i) => [0, 1].forEach(w => {
    const td = rows[i + 1].cells[w + 1], s = G.sh[w]; let txt = '', cls = '';
    if (r.c !== undefined) {
      if (s.f[r.c]) { txt = s.p[r.c]; cls = s.p[r.c] === 0 ? 'zero' : ''; }
      else if (w === 0 && t && myTurn && t.L[di][r.c]) { txt = t.S[di][r.c]; cls = 'pot can'; td.dataset.c = r.c; }
      else txt = '·';
      if (G.flash && G.flash.who === w && G.flash.c === r.c) cls += ' flash';
    } else if (r.k === 'sub') txt = upper(s); else if (r.k === 'bon') txt = upper(s) >= 63 ? 35 : '–';
    else if (r.k === 'yb') txt = s.yb ? 100 * s.yb : '–'; else txt = total(s);
    td.textContent = txt; td.className = cls;
  }));
}
const msg = t => $('msg').innerHTML = '<span>' + t + '</span>';
const addLog = (who, t) => { const li = document.createElement('li'); li.className = who ? 'ai' : ''; li.textContent = t; $('log').prepend(li); while ($('log').children.length > 6) $('log').lastChild.remove(); };

// ---- lancers
async function animRoll(idx) {
  const g = G; g.rolling = idx;
  for (let f = 0; f < 7; f++) { idx.forEach(i => g.dice[i] = 1 + Math.floor(Math.random() * 6)); render(); await sleep(55); if (G !== g) return false; }
  g.rolling = []; render(); return true;
}
function toggle(i) { if (G.turn === 0 && !G.busy && !G.over && G.rolls > 0 && G.rolls < 3) { G.held[i] = !G.held[i]; render(); } }
$('roll').onclick = async () => {
  if (G.busy || G.turn !== 0 || G.rolls === 0) return;
  const idx = G.rolls === 3 ? IDX5 : IDX5.filter(i => !G.held[i]);
  if (!idx.length) { msg('Tous tes dés sont gardés : choisis une case ou libère des dés.'); return; }
  G.busy = true; render(); if (!await animRoll(idx)) return; G.rolls--; G.busy = false; render();
  msg(G.rolls > 0 ? 'Garde des dés (clic) et relance, ou choisis une case <b>bleue</b> dans la feuille.' : 'Plus de relance : choisis une case <b>bleue</b> dans la feuille.');
};
$('card').onclick = e => {
  const c = e.target.dataset && e.target.dataset.c;
  if (c === undefined || !e.target.classList.contains('can') || G.busy || G.turn !== 0) return;
  const d = [...G.dice].sort((a, b) => a - b); apply(G.sh[0], d, +c);
  addLog(0, `Toi : ${NAMES[c]} → ${G.sh[0].p[c]} pts`); endTurn();
};

// ---- déroulement
function startTurn() {
  if (G.sh.every(s => s.f.every(Boolean))) return finish();
  G.dice = [0,0,0,0,0]; G.held = []; G.rolls = 3; G.flash = null; G.busy = false; render();
  if (G.turn === 1) aiTurn(); else msg('🎯 À toi : lance les dés !');
}
function endTurn() { G.turn ^= 1; startTurn(); }

async function aiTurn() {
  const g = G, wait = async ms => { await sleep(ms); return G === g; };
  G.busy = true; const me = G.sh[1], opp = G.sh[0];
  msg('🤖 L’IA réfléchit…'); render(); if (!await wait(700)) return; G.ai.plan(me, opp);
  if (!await animRoll(IDX5)) return; G.rolls = 2; msg(`🤖 L’IA obtient <b>${[...G.dice].sort((a,b)=>a-b).join(' · ')}</b>`); if (!await wait(1000)) return;
  const kept = [];
  for (const r of [2, 1]) {
    const keep = G.ai.chooseKeep(G.dice, r, me, opp); G.held = keep; render();
    const k = G.dice.filter((_, i) => keep[i]);
    if (keep.every(Boolean)) { msg('🤖 L’IA <b>garde tous ses dés</b> et s’arrête.'); if (!await wait(1300)) return; break; }
    msg(`🤖 L’IA garde <b>${k.length ? k.join(' · ') : 'rien'}</b> et relance ${5 - k.length} dé${5 - k.length > 1 ? 's' : ''}`); if (!await wait(1600)) return;
    if (!await animRoll(IDX5.filter(i => !keep[i]))) return; G.rolls = r - 1;
  }
  G.held = []; render();
  const d = [...G.dice].sort((a, b) => a - b), c = G.ai.chooseCategory(d);
  G.flash = { who: 1, c }; apply(me, d, c); render();
  msg(`🤖 L’IA joue <b>${NAMES[c]}</b> : <b>${me.p[c]} pts</b>`); addLog(1, `IA (${d.join('')}) : ${NAMES[c]} → ${me.p[c]} pts`);
  if (!await wait(1800)) return; endTurn();
}

function finish() {
  G.over = true; G.busy = false; render();
  const a = total(G.sh[0]), b = total(G.sh[1]);
  $('endTitle').textContent = a > b ? '🏆 Victoire !' : a < b ? '🤖 L’IA gagne' : '🤝 Égalité';
  $('endText').innerHTML = `Toi : <b>${a}</b> pts &nbsp;·&nbsp; IA : <b>${b}</b> pts`;
  msg(a > b ? 'Bravo, tu as battu le réseau !' : 'Partie terminée.'); $('end').hidden = false;
}

function newGame() {
  if (!window.YAHTZEE_NET) { msg('⚠️ weights.js introuvable : lance <code>python export_weights.py scorenet_best.npz</code>.'); return; }
  const f = $('first').value;
  G = { spd: +$('speed').value, mode: $('mode').value, sh: [newSheet(), newSheet()], turn: f === 'r' ? Math.round(Math.random()) : +f,
        dice: [0,0,0,0,0], held: [], rolls: 3, rolling: [], busy: false, over: false, flash: null };
  G.ai = new AI(G.mode); $('end').hidden = true; $('log').innerHTML = ''; buildCard();
  addLog(G.turn, G.turn ? 'L’IA commence.' : 'Tu commences.'); startTurn();
}
$('new').onclick = $('again').onclick = newGame;
$('speed').onchange = () => G.spd = +$('speed').value;
newGame();
})();
