/* ═══════════════════════════════════════════════════════
   MAZU Dashboard — Frontend Logic
   Vanilla JS, no framework. ES modules not needed.
   ═══════════════════════════════════════════════════════ */

// ── Globals ──
const API = '';
let currentLang = localStorage.getItem('mazu-lang') || 'en';
let sessionId = '';
let activeTab = 'overview';

const REGIONS = [
  { id: 'auto',    en: 'Auto-detect',   ar: 'تلقائي' },
  { id: 'riyadh',  en: 'Riyadh',       ar: 'الرياض' },
  { id: 'jeddah',  en: 'Jeddah',       ar: 'جدة' },
  { id: 'mecca',   en: 'Mecca',        ar: 'مكة' },
  { id: 'dammam',  en: 'Dammam',       ar: 'الدمام' },
  { id: 'jubail',  en: 'Jubail',       ar: 'الجبيل' },
  { id: 'abha',    en: 'Abha',         ar: 'أبها' },
  { id: 'tabuk',   en: 'Tabuk',        ar: 'تبوك' },
];

const HAZARDS = {
  extreme_heat:       { color: '#dc2626', icon: '🔥', en: 'Extreme Heat',        ar: 'حرارة شديدة' },
  dust_storm:         { color: '#ea580c', icon: '💨', en: 'Dust Storm',          ar: 'عاصفة غبارية' },
  flash_flood:        { color: '#2563eb', icon: '🌊', en: 'Flash Flood',         ar: 'سيول مفاجئة' },
  coastal_humid_heat: { color: '#d97706', icon: '🏖️', en: 'Coastal Humid Heat',  ar: 'حرارة ساحلية رطبة' },
};

const QUICK_PROMPTS = [
  { en: 'What hazards does Riyadh face tomorrow?',                 ar: 'ما المخاطر التي تواجه الرياض غداً؟' },
  { en: 'Compare this week heat with last year',                  ar: 'قارن حرارة هذا الأسبوع بالعام الماضي' },
  { en: 'Any dust storm risk in the Persian Gulf?',                ar: 'هل هناك خطر عاصفة غبارية في الخليج؟' },
  { en: 'Flash flood risk in the Asir mountains?',                 ar: 'خطر السيول في جبال عسير؟' },
];

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  setLang(currentLang);
  setupTabs();
  setupSidebar();
  renderRegionPills();
  renderQuickPrompts();
  checkServices();
  setupScrollReveal();
  setupCardGlow();
});

// ── Language ──
function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('mazu-lang', lang);
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-en][data-ar]').forEach(el => {
    el.textContent = lang === 'ar' ? el.dataset.ar : el.dataset.en;
  });
  document.querySelectorAll('input[placeholder]').forEach(el => {
    if (el.dataset.en && el.dataset.ar) {
      el.placeholder = lang === 'ar' ? el.dataset.ar : el.dataset.en;
    }
  });
  document.getElementById('btn-en').classList.toggle('active', lang === 'en');
  document.getElementById('btn-ar').classList.toggle('active', lang === 'ar');
}

function t(en, ar) { return currentLang === 'ar' && ar ? ar : en; }

// ── Tabs ──
function setupTabs() {
  document.querySelectorAll('.tab-btn, .sidebar-btn[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.sidebar-btn[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  const target = document.getElementById('tab-' + tab);
  if (target) target.style.display = 'block';
}

function setupSidebar() {}

// ── Region pills ──
function renderRegionPills() {
  const container = document.getElementById('region-pills');
  if (!container) return;
  container.innerHTML = REGIONS.map(r =>
    `<span class="region-pill" data-region="${r.id}" onclick="selectRegion('${r.id}')">${currentLang === 'ar' ? r.ar : r.en}</span>`
  ).join('');
}

function selectRegion(id) {
  document.querySelectorAll('.region-pill').forEach(p => p.classList.toggle('active', p.dataset.region === id));
  document.getElementById('city-input').value = id === 'auto' ? '' : id;
}

// ── Quick prompts ──
function renderQuickPrompts() {
  const container = document.getElementById('quick-prompts');
  if (!container) return;
  container.innerHTML = QUICK_PROMPTS.map((p, i) =>
    `<span class="quick-prompt" onclick="sendQuickPrompt(${i})">${currentLang === 'ar' ? p.ar : p.en}</span>`
  ).join('');
}

function sendQuickPrompt(i) {
  switchTab('chat');
  const msg = currentLang === 'ar' ? QUICK_PROMPTS[i].ar : QUICK_PROMPTS[i].en;
  sendChatMessage(msg);
}

// ── Service status ──
async function checkServices() {
  try {
    const r = await fetch(API + '/api/kg/summary');
    setServiceStatus('kg', r.ok);
  } catch { setServiceStatus('kg', false); }
  try {
    const r = await fetch(API + '/api/session/list');
    setServiceStatus('agent', r.ok);
  } catch { setServiceStatus('agent', false); }
}

function setServiceStatus(svc, online) {
  const el = document.getElementById('status-' + svc);
  if (!el) return;
  el.className = 'sidebar-status ' + (online ? 'online' : 'offline');
  const dot = el.querySelector('.status-dot');
  if (dot) dot.className = 'status-dot ' + (online ? 'online' : 'offline');
}

// ── City detection ──
async function runCityDetection() {
  const city = document.getElementById('city-input').value.trim() || 'all major Saudi cities';
  const overview = document.getElementById('overview-results');
  overview.innerHTML = '<div class="card"><div class="spinner"></div> Running detection for <strong>' + city + '</strong>...</div>';

  try {
    const msg = 'Run get_calibrated_city_hazards for ' + city + ' and report the results with severity, historical base rate, and calibrated confidence for all four hazard types.';
    const r = await fetch(API + '/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });
    const data = await r.json();
    sessionId = data.session_id || sessionId;
    overview.innerHTML = renderAgentReply(data.reply);
    document.getElementById('stat-time').textContent = new Date().toLocaleTimeString();
  } catch (e) {
    overview.innerHTML = '<div class="card" style="color:var(--red)">Error: ' + e.message + '</div>';
  }
}

// ── Single hazard detection ──
async function runHazardDetect(htype, inputId, resultId) {
  const city = document.getElementById(inputId).value.trim();
  if (!city) return;
  const resultEl = document.getElementById(resultId);
  resultEl.innerHTML = '<div class="card"><div class="spinner"></div> Detecting ' + HAZARDS[htype].en + ' for ' + city + '...</div>';

  try {
    const hazardLabel = HAZARDS[htype].en;
    const msg = 'Run get_calibrated_city_hazards for ' + city + ' and report only the ' + hazardLabel + ' results. Format: severity (PXX, base rate X%, confidence: high/medium/low).';
    const r = await fetch(API + '/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });
    const data = await r.json();
    sessionId = data.session_id || sessionId;
    resultEl.innerHTML = renderAgentReply(data.reply);
  } catch (e) {
    resultEl.innerHTML = '<div class="card" style="color:var(--red)">Error: ' + e.message + '</div>';
  }
}

// ── Chat ──
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  sendChatMessage(msg);
}

async function sendChatMessage(msg) {
  const history = document.getElementById('chat-history');
  history.innerHTML += '<div class="chat-msg user"><div class="bubble">' + escapeHtml(msg) + '</div></div>';

  const thinkingId = 'thinking-' + Date.now();
  history.innerHTML += '<div class="chat-msg assistant" id="' + thinkingId + '"><div class="bubble"><span class="spinner"></span> Thinking...</div></div>';
  history.scrollTop = history.scrollHeight;

  try {
    const r = await fetch(API + '/api/chat/stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: sessionId }),
    });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let hasToolCalls = false;
    const thinkingEl = document.getElementById(thinkingId);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const evt = JSON.parse(line.slice(6));
          if (evt.type === 'text') {
            fullContent += evt.content;
            if (!hasToolCalls && thinkingEl) {
              thinkingEl.querySelector('.bubble').innerHTML = '<span class="spinner"></span> Thinking...';
            }
          } else if (evt.type === 'tool_calls') {
            hasToolCalls = true;
            fullContent = '';
            if (thinkingEl) thinkingEl.querySelector('.bubble').innerHTML = '<span class="spinner"></span> <small style="color:var(--textMuted)">Calling tools: ' + evt.calls.map(c => c.name).join(', ') + '</small>';
          } else if (evt.type === 'done') {
            if (thinkingEl) thinkingEl.querySelector('.bubble').innerHTML = renderMarkdown(fullContent);
          } else if (evt.type === 'error') {
            if (thinkingEl) thinkingEl.querySelector('.bubble').innerHTML = '<span style="color:var(--red)">Error: ' + evt.content + '</span>';
          }
        } catch {}
      }
    }
  } catch (e) {
    document.getElementById(thinkingId).querySelector('.bubble').innerHTML = '<span style="color:var(--red)">Connection error: ' + e.message + '</span>';
  }
  document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
}

// ── Render helpers ──
function renderAgentReply(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'card reveal visible';
  wrapper.innerHTML = '<div class="hazard-detail">' + renderMarkdown(text) + '</div>';
  return wrapper.outerHTML;
}

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  html = html.replace(/\|(.+)\|\n\|[-| ]+\|\n((?:\|.+\|\n?)*)/g, (match, header, rows) => {
    const hcells = header.split('|').filter(c => c.trim()).map(c => '<th>' + c.trim() + '</th>').join('');
    const rrows = rows.trim().split('\n').map(row =>
      '<tr>' + row.split('|').filter(c => c.trim()).map(c => '<td>' + c.trim() + '</td>').join('') + '</tr>'
    ).join('');
    return '<table><thead><tr>' + hcells + '</tr></thead><tbody>' + rrows + '</tbody></table>';
  });

  html = '<p>' + html + '</p>';
  html = html.replace(/<p><\/p>/g, '');
  return html;
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ═══════════════════════════════════════════════════════
// SCROLL REVEAL — IntersectionObserver for .reveal
// ═══════════════════════════════════════════════════════
function setupScrollReveal() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });

  document.querySelectorAll('.stat-tile, .hazard-card, .card, .stats-row > div').forEach(el => {
    el.classList.add('reveal');
    observer.observe(el);
  });

  const tabObserver = new MutationObserver(() => {
    document.querySelectorAll('.tab-content:not([style*="display: none"]) .card, .tab-content:not([style*="display: none"]) .stat-tile').forEach(el => {
      if (!el.classList.contains('reveal')) {
        el.classList.add('reveal');
        observer.observe(el);
      }
    });
  });
  document.querySelectorAll('.tab-content').forEach(el => {
    tabObserver.observe(el, { attributes: true, attributeFilter: ['style'] });
  });
}

// ═══════════════════════════════════════════════════════
// CARD GLOW — magnetic light follows cursor on .card
// ═══════════════════════════════════════════════════════
function setupCardGlow() {
  document.addEventListener('mousemove', (e) => {
    document.querySelectorAll('.card:hover, .stat-tile:hover').forEach(card => {
      const rect = card.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;
      card.style.setProperty('--mx', x + '%');
      card.style.setProperty('--my', y + '%');
    });
  }, { passive: true });
}
