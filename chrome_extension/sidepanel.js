/**
 * sidepanel.js — final updated version
 * - Colored rdot per category (high/medium/low)
 * - Prefer explicit JSON fields for justification/user_rights
 * - Show all clauses in Summary (muted note when missing)
 * - Deduplicate conservatively using full normalized text
 * - Increased summary font sizes, removed confidence and Read more
 * - Interactive expand/collapse for findings and summary rows
 *
 * Replace your existing sidepanel.js with this file (no other edits needed).
 */

const API_URL = "http://localhost:8080/analyze";

// ── State & DOM ───────────────────────────────────────────────
let g = { url: '', domain: '', report: null, scanning: false };

const hdrDom   = document.getElementById('hdr-dom');
const riskbar  = document.getElementById('riskbar');
const rpill    = document.getElementById('rpill');
const sh       = document.getElementById('sh');
const sm       = document.getElementById('sm');
const stot     = document.getElementById('stot');
const cdot     = document.getElementById('cdot');
const csrc     = document.getElementById('csrc');
const ftrHint  = document.getElementById('ftr-hint');
const btnScan  = document.getElementById('btn-scan');
const btnSum   = document.getElementById('btn-summary');
const pf       = document.getElementById('pf'); // findings panel
const ps       = document.getElementById('ps'); // summary panel
const pp       = document.getElementById('pp');
const tabSum   = document.getElementById('tab-sum');

// ── Tabs ─────────────────────────────────────────────────────
document.getElementById('tabs').addEventListener('click', e => {
  const tab = e.target.closest('.tab');
  if (!tab || tab.classList.contains('off')) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('on'));
  tab.classList.add('active');
  document.getElementById(tab.dataset.p).classList.add('on');
});

// ── Boot ─────────────────────────────────────────────────────
(async function boot() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url?.startsWith('http')) {
    sbox(pf, 'Cannot scan this page', 'Navigate to a website first.');
    ftrHint.textContent = 'Unsupported page';
    return;
  }
  g.url    = tab.url;
  g.domain = extractDomain(tab.url);
  hdrDom.textContent   = g.domain;
  btnScan.style.display = 'inline-flex';

  chrome.scripting.executeScript({ target: { tabId: tab.id }, func: auditPerms });
  await checkCache();
})();

// ── Cache check ───────────────────────────────────────────────
async function checkCache() {
  ftrHint.textContent = 'Checking database...';
  sbox(pf, 'Checking database', 'Looking up ' + g.domain + '...');
  try {
    const r = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: g.url, chunks: [], cache_only: true }),
    });
    if (r.ok) {
      const data = await r.json();
      if (data?.summary) { g.report = data; render(data, 'cached'); return; }
    }
  } catch (_) {}
  sbox(pf, 'No cached data for this domain',
    'Click "Scan Policy" on a privacy policy page to classify its clauses.');
  ftrHint.textContent = 'No data found';
}

// ── Scan button ───────────────────────────────────────────────
btnScan.addEventListener('click', async () => {
  if (g.scanning) return;
  g.scanning = true;
  btnScan.disabled = true;
  btnScan.textContent = 'Scanning...';
  btnSum.disabled = true;

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.storage.local.remove(['pageChunks', 'pageUrl']);

    sbox(pf, 'Extracting text', 'Reading paragraphs from the page...');
    ftrHint.textContent = 'Extracting...';

    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    const chunks = await waitChunks(7000);

    if (!chunks?.length) {
      sbox(pf, 'No text found', 'Navigate to the actual privacy policy page and try again.');
      ftrHint.textContent = 'Extraction failed';
      return;
    }

    sbox(pf, `Classifying ${chunks.length} clauses`, 'LegalBERT is reading each clause...');
    ftrHint.textContent = 'Classifying...';

    const r = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: g.url, chunks }),
    });
    if (!r.ok) throw new Error('API ' + r.status);
    const report = await r.json();

    g.report = report;
    render(report, 'live');
    chrome.storage.local.set({ riskReport: report });

  } catch (err) {
    sbox(pf, 'Scan failed', err.message);
    ftrHint.textContent = 'Error';
  } finally {
    g.scanning = false;
    btnScan.disabled = false;
    btnScan.textContent = 'Scan Policy';
  }
});

// ── Summary button ─────────────────────────────────────────────
btnSum.addEventListener('click', () => {
  if (!g.report) return;
  document.querySelector('[data-p="ps"]').click();
  buildSummary();
});

// ── Render findings ───────────────────────────────────────────
function render(report, src) {
  const s = report.summary || {};
  const overall = s.overall_risk || 'UNKNOWN';

  riskbar.style.display = 'flex';
  rpill.textContent = overall;
  rpill.className   = 'rpill ' + overall;
  sh.textContent    = s.high_risk_clauses   ?? 0;
  sm.textContent    = s.medium_risk_clauses ?? 0;
  stot.textContent  = s.total_clauses       ?? 0;
  cdot.className    = 'cdot ' + (src === 'live' ? 'l' : 'c');
  csrc.textContent  = src === 'live' ? 'Live scan' : 'Cached';

  const all = dedupeFindings([...(report.high_risk_findings || []), ...(report.medium_risk_findings || []), ...(report.low_risk_findings || [])]);
  if (!all.length) {
    sbox(pf, 'All clear', 'No findings detected.');
    ftrHint.textContent = 'No risks found';
    return;
  }

  tabSum.classList.remove('off');
  btnSum.style.display = 'inline-flex';
  btnSum.disabled      = false;

  pf.innerHTML = '';
  const groups = groupBy(all);
  for (const [cat, clauses] of Object.entries(groups)) {
    pf.appendChild(catBlock(cat, clauses));
  }
  ftrHint.textContent = `${all.length} findings — ${src === 'live' ? 'live scan' : 'cached'}`;
}

// ── Determine category risk (highest risk among clauses) ──────
function getCategoryRisk(clauses) {
  if (!clauses || !clauses.length) return 'low';
  if (clauses.some(c => (c.risk_level || '').toLowerCase() === 'high')) return 'high';
  if (clauses.some(c => (c.risk_level || '').toLowerCase() === 'medium')) return 'medium';
  return 'low';
}

// ── Build category block (findings list) ──────────────────────
function catBlock(cat, clauses) {
  const wrap = el('div', 'cat-block');

  const hdr = el('div', 'cat-hdr');
  const catRisk = getCategoryRisk(clauses);
  hdr.innerHTML = `<span class="rdot ${catRisk}"></span><span class="cat-name">${esc(cat)}</span><span class="cat-cnt">${clauses.length} clause${clauses.length > 1 ? 's' : ''}</span>`;
  wrap.appendChild(hdr);

  clauses.forEach((f, i) => {
    const row  = el('div', 'clause');
    // interactive row styles (kept inline to avoid changing CSS files)
    row.style.padding = '10px 8px';
    row.style.borderBottom = '1px solid rgba(255,255,255,0.03)';
    row.style.cursor = 'pointer';
    row.style.transition = 'background 160ms ease';
    row.addEventListener('mouseenter', () => row.style.background = 'rgba(255,255,255,0.02)');
    row.addEventListener('mouseleave', () => row.style.background = 'transparent');

    const textEl = el('div', 'ctext');
    textEl.textContent = f.clause_text || '';
    // use CSS class toggle to expand/collapse (consistent with stylesheet)
    textEl.classList.remove('open');
    textEl.style.fontSize = '14px';
    textEl.style.lineHeight = '1.35';
    textEl.style.marginBottom = '6px';

    // clicking the row toggles expansion
    row.addEventListener('click', () => {
      const isOpen = textEl.classList.toggle('open');
      row.setAttribute('data-expanded', isOpen ? '1' : '0');
    });

    const meta = el('div', 'cmeta');
    meta.style.fontSize = '12px';
    meta.style.color = '#9aa0a6';
    // confidence display removed per request

    const body = el('div', 'cbody');
    body.appendChild(textEl);
    body.appendChild(meta);

    row.innerHTML = `<span class="cnum" style="display:inline-block;width:22px;font-weight:700;color:#9aa0a6">${i + 1}.</span>`;
    row.appendChild(body);
    wrap.appendChild(row);
  });

  return wrap;
}

// ── Build summary panel (clean + presentable + interactive) ───
function buildSummary() {
  const all = dedupeFindings([...(g.report.high_risk_findings || []), ...(g.report.medium_risk_findings || []), ...(g.report.low_risk_findings || [])]);
  if (!all.length) { sbox(ps, 'Nothing to summarise', 'No findings available.'); return; }

  ps.innerHTML = '';
  const groups = groupBy(all);

  for (const [cat, clauses] of Object.entries(groups)) {
    const card = el('div', 'sum-card');
    card.style.padding = '12px';
    card.style.marginBottom = '12px';
    card.style.borderRadius = '6px';
    card.style.background = '#0f1113';
    card.style.transition = 'box-shadow 160ms ease';

    const catRisk = getCategoryRisk(clauses);
    const hdr = el('div', 'sum-hdr');
    hdr.innerHTML = `<span class="rdot ${catRisk}"></span><span class="cat-name" style="font-weight:700;font-size:15px">${esc(cat)}</span><span class="cat-cnt" style="float:right;color:#9aa0a6;font-size:12px">${clauses.length} clause${clauses.length > 1 ? 's' : ''}</span>`;
    hdr.style.marginBottom = '8px';
    card.appendChild(hdr);

    const body = el('div', 'sum-body');

    let shown = 0;
    clauses.forEach((c, i) => {
      const expl = c.explanation || '';

      // Prefer explicit JSON fields if present
      const justification = (
        (c.justification || c.justification_text || (c.summary && c.summary.justification)) ||
        extractJustification(expl) || ''
      ).toString().trim();

      const userRights = (
        (c.user_rights || c.userRights || (c.summary && c.summary.user_rights)) ||
        extractUserRights(expl) || ''
      ).toString().trim();

      shown += 1;

      const shortClause = (c.clause_text || pullExcerpt(c.explanation) || '').trim().replace(/\s+/g, ' ');
      const shortLabel = shortClause.length > 220 ? shortClause.slice(0, 217) + '...' : (shortClause || ('Clause ' + (i + 1)));

      const row = el('div', 'ex-row');
      row.style.padding = '10px 0';
      row.style.borderBottom = '1px solid rgba(255,255,255,0.03)';
      row.style.lineHeight = '1.45';
      row.style.cursor = 'pointer';
      row.addEventListener('mouseenter', () => row.style.background = 'rgba(255,255,255,0.02)');
      row.addEventListener('mouseleave', () => row.style.background = 'transparent');

      // numbering label
      const label = el('div');
      label.innerHTML = `<span style="font-weight:700;color:#e8eef6;font-size:15px">${shown}. ${toHtml(escText(shortLabel))}</span>`;
      label.style.marginBottom = '8px';
      row.appendChild(label);

      // justification (show full but visually clamped; clicking expands)
      const j = el('div');
      j.className = 'sum-just';
      if (justification) {
        j.innerHTML = `<b style="font-weight:600">Justification:</b> ${toHtml(escText(justification))}`;
      } else {
        j.innerHTML = `<i style="color:#8f979b">Justification: — no justification extracted</i>`;
      }
      j.style.display = '-webkit-box';
      j.style.webkitBoxOrient = 'vertical';
      j.style.webkitLineClamp = '2';
      j.style.overflow = 'hidden';
      j.style.color = '#bfc7cc';
      j.style.fontSize = '14px';
      j.style.marginBottom = '6px';
      row.appendChild(j);

      // user rights
      const u = el('div');
      u.className = 'sum-rights';
      if (userRights) {
        u.innerHTML = `<b style="font-weight:600">User rights:</b> ${toHtml(escText(userRights))}`;
      } else {
        u.innerHTML = `<i style="color:#8f979b">User rights: — not found</i>`;
      }
      u.style.display = '-webkit-box';
      u.style.webkitBoxOrient = 'vertical';
      u.style.webkitLineClamp = '2';
      u.style.overflow = 'hidden';
      u.style.color = '#bfc7cc';
      u.style.fontSize = '14px';
      row.appendChild(u);

      // clicking toggles expanded state
      row.addEventListener('click', () => {
        const expanded = row.getAttribute('data-expanded') === '1';
        if (expanded) {
          j.style.webkitLineClamp = '2';
          u.style.webkitLineClamp = '2';
          row.setAttribute('data-expanded', '0');
          row.style.boxShadow = 'none';
        } else {
          j.style.webkitLineClamp = 'unset';
          u.style.webkitLineClamp = 'unset';
          row.setAttribute('data-expanded', '1');
          row.style.boxShadow = '0 6px 18px rgba(0,0,0,0.35)';
        }
      });

      body.appendChild(row);
    });

    if (body.childElementCount === 0) continue;

    card.appendChild(body);
    ps.appendChild(card);
  }

  if (!ps.childElementCount) {
    sbox(ps, 'Nothing to summarise', 'No clauses contained justification/user-rights summaries.');
  }
}

// ── Deduplicate repeated findings by normalized clause_text (conservative) ───
function dedupeFindings(findings) {
  const seen = new Set();
  const out = [];
  for (const f of findings) {
    const text = (f.clause_text || f.explanation || '').replace(/\s+/g, ' ').trim().toLowerCase();
    if (!text) continue;
    const key = text; // full normalized text avoids accidental collisions
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(f);
  }
  return out;
}

// ── Text helpers & extractors ────────────────────────────────

function cleanExp(text) {
  if (!text) return '';
  return text
    .replace(/(?:(?:Risk Level)[,.]?\s*){3,}/gi, '')
    .replace(/\*\*Data Details:\*\*[\s\S]*?(?=\n\n|\n\*\*|$)/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function stripClauseClassification(text) {
  if (!text) return '';
  return text
    .replace(/\*\*\s*Clause Classification:\s*\*\*/gi, '')
    .replace(/\*\*Clause Classification:\*\*/gi, '')
    .replace(/Clause Classification:\s*/gi, '')
    .replace(/\*\*Clause:\*\*/gi, '')
    .replace(/\*\*Clause\s*\d+:\*\*/gi, '')
    .replace(/\*\*Clause\s*\d+\.\*\*/gi, '')
    .replace(/\*\*Clause\s*/gi, '')
    .replace(/\*\*Clause\*\*/gi, '')
    .trim();
}

function extractJustification(text) {
  if (!text) return '';
  let t = cleanExp(text);
  t = stripClauseClassification(t);

  const patterns = [
    /Justification(?: for (?:the )?Risk Level)?:\s*([\s\S]*?)(?=\n\n|\n\*\*|$)/i,
    /\*\*Analysis:\*\*\s*([\s\S]*?)(?=\n\n|\n\*\*|$)/i,
    /Analysis:\s*([\s\S]*?)(?=\n\n|\n\*\*|$)/i,
    /Risk Assessment:\s*([\s\S]*?)(?=\n\n|\n\*\*|$)/i,
    /(This clause falls under[^\n\r]+)/i
  ];

  for (const p of patterns) {
    const m = t.match(p);
    if (m && m[1]) {
      const out = m[1].trim();
      return stripClauseClassification(out);
    }
  }

  const para = t.split(/\n\n/)[0].trim();
  if (para.length) return stripClauseClassification(para);
  return '';
}

function extractUserRights(text) {
  if (!text) return '';
  let t = cleanExp(text);
  t = stripClauseClassification(t);

  const patterns = [
    /User Rights:\s*([\s\S]*?)(?=\n\n|\n\*\*|$)/i,
    /Users can ([^\.\n]+)/i,
    /(Users can opt[- ]out[^\.\n]+)/i,
    /(Limited;[^\n]+)/i
  ];

  for (const p of patterns) {
    const m = t.match(p);
    if (m && m[1]) return stripClauseClassification(m[1].trim());
  }

  const m2 = t.match(/(opt-?out[^.\n]*|limited control[^.\n]*|cannot opt[^.\n]*|rights to [^\.\n]+)/i);
  if (m2) return stripClauseClassification(m2[1].trim());

  return '';
}

function pullExcerpt(text) {
  if (!text) return '';
  const m = text.match(/Key Excerpt[s]?:\*\*\s*"([^"\n]+)"/i);
  return m ? m[1].trim() : '';
}

function groupBy(findings) {
  const g = {};
  [...findings]
    .sort((a, b) => {
      if ((a.risk_level || '') !== (b.risk_level || '')) return (a.risk_level === 'high') ? -1 : 1;
      return (a.clause_type || '').localeCompare(b.clause_type || '');
    })
    .forEach(f => { const c = (f.clause_type || 'Other'); (g[c] = g[c] || []).push(f); });

  // dedupe identical clause_texts inside each category (should already be deduped globally)
  Object.keys(g).forEach(k => {
    const seen = new Set();
    g[k] = g[k].filter(f => {
      const txt = (f.clause_text || f.explanation || '').replace(/\s+/g, ' ').trim().toLowerCase();
      if (!txt) return false;
      if (seen.has(txt)) return false;
      seen.add(txt);
      return true;
    });
  });

  return g;
}

function waitChunks(ms) {
  return new Promise(res => {
    const t0 = Date.now();
    (function check() {
      chrome.storage.local.get('pageChunks', ({ pageChunks }) => {
        if (pageChunks?.length) return res(pageChunks);
        if (Date.now() - t0 > ms) return res(null);
        setTimeout(check, 200);
      });
    })();
  });
}

function sbox(panel, title, sub) {
  panel.innerHTML = `<div class="sbox"><div class="st">${esc(title)}</div><div class="ss">${esc(sub)}</div></div>`;
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function extractDomain(url) {
  try { let d = new URL(url).hostname; return d.startsWith('www.') ? d.slice(4) : d; }
  catch { return url; }
}

function esc(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// allow limited markdown to render (bold/italic) safely
function toHtml(text) {
  if (!text) return '';
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\*([^*\n]+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

function escText(s) {
  if (!s) return '';
  return s.replace(/\*\*\s*Clause Classification:\s*\*\*/gi, '')
          .replace(/\*\*Clause Classification:\*\*/gi, '')
          .replace(/\*\*Clause:\*\*/gi, '')
          .replace(/\*\*Clause\*\*/gi, '')
          .trim();
}

// ── Permission auditor (injected into page) ───────────────────
function auditPerms() {
  const ps = ['geolocation', 'camera', 'microphone', 'notifications'];
  const granted = [];
  Promise.allSettled(
    ps.map(p => navigator.permissions.query({ name: p })
      .then(s => { if (s.state === 'granted') granted.push(p); })
      .catch(() => {}))
  ).then(() => chrome.runtime.sendMessage({ type: 'PERM', granted }));
}

chrome.runtime.onMessage.addListener(msg => {
  if (msg.type !== 'PERM') return;
  pp.innerHTML = '';
  const c = el('div', 'perm-card ' + (msg.granted.length ? 'bad' : 'ok'));
  if (msg.granted.length) {
    c.innerHTML = `
      <div class="perm-title">Hardware permissions active</div>
      <div class="perm-desc">This site currently has browser permission to access your:</div>
      <div class="perm-list">${msg.granted.join('<br>')}</div>
      <div class="perm-desc" style="margin-top:9px">Click the padlock in the URL bar, then Site settings, to revoke.</div>`;
  } else {
    c.innerHTML = `
      <div class="perm-title">No hardware permissions granted</div>
      <div class="perm-desc">This site cannot access your camera, microphone, or location.</div>`;
  }
  pp.appendChild(c);
});