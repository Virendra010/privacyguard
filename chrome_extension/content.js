/**
 * content.js v3
 * Extracts text chunks, scrubs PII, stores in chrome.storage.
 * Also listens for highlight instructions from the side panel.
 */

const PII_PATTERNS = [
  { p: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, r: '[EMAIL]' },
  { p: /\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b/g,                    r: '[PHONE]' },
  { p: /[?&]utm_[a-z_]+=\S*/gi,                                  r: '' },
  { p: /[?&]fbclid=\S*/gi,                                       r: '' },
];

function scrub(text) {
  let t = text;
  for (const { p, r } of PII_PATTERNS) t = t.replace(p, r);
  return t.trim();
}

// Build element map for highlight restoration
const elementMap = new Map();

function extractChunks() {
  const els    = document.querySelectorAll('p, li, section > div');
  const chunks = [];
  let   id     = 0;

  els.forEach(el => {
    const text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
    if (text.length > 80) {
      elementMap.set(id, el);
      chunks.push({ id: id++, text: scrub(text) });
    }
  });
  return chunks;
}

function highlightFromReport(report) {
  // Reset
  elementMap.forEach(el => {
    el.style.backgroundColor = '';
    el.style.borderLeft      = '';
    el.style.paddingLeft     = '';
  });

  const all = [
    ...(report.high_risk_findings   || []),
    ...(report.medium_risk_findings || []),
  ];
  all.forEach(f => {
    const el = elementMap.get(f.chunk_id);
    if (!el) return;
    el.style.backgroundColor = f.risk_level === 'high' ? '#2d1b1b' : '#2d2618';
    el.style.borderLeft      = f.risk_level === 'high' ? '3px solid #f85149' : '3px solid #e3b341';
    el.style.paddingLeft     = '6px';
  });
}

// Run
const chunks = extractChunks();
chrome.storage.local.set({ pageChunks: chunks, pageUrl: window.location.href });

// Listen for highlight instructions
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.riskReport?.newValue) {
    highlightFromReport(changes.riskReport.newValue);
  }
});
