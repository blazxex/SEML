// Stock Sentiment Dashboard — charts + day-detail panel

let sentimentChart    = null;
let distributionChart = null;
let gaugeChart        = null;
let _currentDates     = [];

const COLORS = {
    buy:       '#3fb950',
    sell:      '#f85149',
    hold:      '#e3b341',
    noOpinion: '#6e7681',
    sentiment: '#639cff',
    rolling3:  '#64dcf0',
    rolling7:  '#b464f0',
    price:     '#ff9f40',
    grid:      '#2a2a4a',
    tick:      '#8080a0',
};

// ── Chart builders ─────────────────────────────────────────────────

function buildSentimentChart(labels, sentData, roll3, roll7, priceData, smoothing, ticker) {
    _currentDates = labels;
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    if (sentimentChart) sentimentChart.destroy();

    const sentDataset = smoothing === '3day'
        ? { label:'3-Day Avg Sentiment', data:roll3, borderColor:COLORS.rolling3, backgroundColor:'rgba(100,220,240,.1)' }
        : smoothing === '7day'
        ? { label:'7-Day Avg Sentiment', data:roll7, borderColor:COLORS.rolling7, backgroundColor:'rgba(180,100,240,.1)' }
        : { label:'Sentiment Score',     data:sentData, borderColor:COLORS.sentiment, backgroundColor:'rgba(100,149,255,.1)' };

    sentimentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                { ...sentDataset, yAxisID:'ySentiment', tension:.3, pointRadius:3, borderWidth:2,
                  pointHoverRadius:6, pointHoverBackgroundColor:sentDataset.borderColor },
                { label:'Close Price ($)', data:priceData, borderColor:COLORS.price,
                  backgroundColor:'rgba(255,159,64,.07)', yAxisID:'yPrice', tension:.3,
                  pointRadius:2, borderWidth:2, pointHoverRadius:5 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode:'index', intersect:false },
            plugins: {
                legend: { labels:{ color:'#c0c0e0' } },
                tooltip: {
                    callbacks: {
                        footer: () => '  Click to open day summary',
                    },
                    footerColor: '#8b949e',
                    footerFont: { style:'italic', size:11 },
                },
            },
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    openDayPanel(ticker || getTicker(), _currentDates[idx]);
                }
            },
            onHover: (evt, elements) => {
                evt.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
            },
            scales: {
                x: { ticks:{ color:COLORS.tick, maxTicksLimit:10 }, grid:{ color:COLORS.grid } },
                ySentiment: {
                    type:'linear', position:'left',
                    ticks:{ color:sentDataset.borderColor },
                    grid:{ color:COLORS.grid },
                    title:{ display:true, text:'Sentiment Score', color:sentDataset.borderColor },
                },
                yPrice: {
                    type:'linear', position:'right',
                    ticks:{ color:COLORS.price },
                    grid:{ drawOnChartArea:false },
                    title:{ display:true, text:'Close Price ($)', color:COLORS.price },
                },
            },
        },
    });
}

function buildDistributionChart(labels, buyPct, holdPct, sellPct, noPct, ticker) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    if (distributionChart) distributionChart.destroy();

    distributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label:'Buy',        data:buyPct.map(v=>+(v*100).toFixed(1)),  backgroundColor:COLORS.buy,       stack:'s' },
                { label:'Hold',       data:holdPct.map(v=>+(v*100).toFixed(1)), backgroundColor:COLORS.hold,      stack:'s' },
                { label:'Sell',       data:sellPct.map(v=>+(v*100).toFixed(1)), backgroundColor:COLORS.sell,      stack:'s' },
                { label:'No Opinion', data:noPct.map(v=>+(v*100).toFixed(1)),   backgroundColor:COLORS.noOpinion, stack:'s' },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend:{ labels:{ color:'#c0c0e0' } },
                tooltip: {
                    callbacks: {
                        footer: () => '  Click to open day summary',
                    },
                    footerColor: '#8b949e',
                    footerFont: { style:'italic', size:11 },
                },
            },
            onClick: (evt, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    openDayPanel(ticker || getTicker(), labels[idx]);
                }
            },
            onHover: (evt, elements) => {
                evt.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
            },
            scales: {
                x: { stacked:true, ticks:{ color:COLORS.tick, maxTicksLimit:10 }, grid:{ color:COLORS.grid } },
                y: { stacked:true, max:100, ticks:{ color:COLORS.tick },
                     grid:{ color:COLORS.grid }, title:{ display:true, text:'%', color:COLORS.tick } },
            },
        },
    });
}

// ── Day Panel ──────────────────────────────────────────────────────

function getTicker() { return document.getElementById('tickerSelect').value; }
function getModel()  { return document.getElementById('modelSelect').value;  }

let _dayPanelOffcanvas = null;
let _tweetPanelOffcanvas = null;

async function openDayPanel(ticker, date) {
    const body = document.getElementById('dayPanelBody');
    document.getElementById('dpTicker').textContent = '…';
    document.getElementById('dpDate').textContent   = date;
    body.innerHTML = '<div class="text-secondary text-center py-5">Loading…</div>';

    if (!_dayPanelOffcanvas) {
        _dayPanelOffcanvas = new bootstrap.Offcanvas(document.getElementById('dayPanel'));
    }
    _dayPanelOffcanvas.show();

    // Open the tweet list on the left in parallel
    openTweetPanel(ticker, date);

    try {
        const res = await fetch(`/api/day?ticker=${ticker}&date=${date}`);
        if (!res.ok) {
            const d = await res.json();
            body.innerHTML = `<div class="text-danger text-center py-5 px-3">${d.error || 'No data for this date.'}</div>`;
            return;
        }
        const data = await res.json();
        renderDayPanel(data);
    } catch(e) {
        body.innerHTML = `<div class="text-danger text-center py-5 px-3">Error: ${e.message}</div>`;
    }
}

async function openTweetPanel(ticker, date) {
    const body = document.getElementById('tweetPanelBody');
    document.getElementById('tpTicker').textContent = `${ticker} — Tweets`;
    document.getElementById('tpDate').textContent   = date;
    document.getElementById('tpCount').textContent  = '';
    body.innerHTML = '<div class="text-secondary text-center py-5">Loading…</div>';

    if (!_tweetPanelOffcanvas) {
        _tweetPanelOffcanvas = new bootstrap.Offcanvas(document.getElementById('tweetPanel'));
    }
    _tweetPanelOffcanvas.show();

    try {
        const res = await fetch(`/api/tweets?ticker=${ticker}&date=${date}&limit=50`);
        if (!res.ok) {
            const d = await res.json();
            body.innerHTML = `<div class="text-danger text-center py-5 px-3">${d.error || 'No tweets.'}</div>`;
            return;
        }
        const data = await res.json();
        renderTweetPanel(data);
    } catch(e) {
        body.innerHTML = `<div class="text-danger text-center py-5 px-3">Error: ${e.message}</div>`;
    }
}

function renderTweetPanel(data) {
    document.getElementById('tpCount').textContent =
        `${data.count}${data.total_for_day && data.total_for_day > data.count ? ' / ' + data.total_for_day : ''} tweets`;

    if (!data.tweets || data.tweets.length === 0) {
        document.getElementById('tweetPanelBody').innerHTML =
            '<div class="text-secondary text-center py-5">No tweets for this date.</div>';
        return;
    }

    const labelColor = (lbl) => ({
        'Buy':        '#3fb950',
        'Sell':       '#f85149',
        'Hold':       '#e3b341',
        'No Opinion': '#8b949e',
    }[lbl] || '#8b949e');

    const html = data.tweets.map((t) => {
        const preds = Object.entries(t.predictions || {})
            .map(([m, lbl]) => `<span class="badge me-1 mb-1" style="background:${labelColor(lbl)};color:#0d1117;font-weight:600;">${m}: ${lbl}</span>`)
            .join('');
        const escTweet = t.tweet
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `
            <div class="tweet-item mb-2 p-2" style="background:#161b22;border:1px solid #30363d;border-radius:6px;">
                <div style="font-size:.85rem;line-height:1.45;color:#e6edf3;">${escTweet}</div>
                <div class="mt-2" style="font-size:.7rem;">${preds}</div>
                <div class="mt-1 d-flex gap-1">
                    <button class="btn btn-success btn-sm py-0 px-1" style="font-size:.7rem;" onclick="quickLabel(${t.tweet_id}, 'Buy', this)">Buy</button>
                    <button class="btn btn-danger  btn-sm py-0 px-1" style="font-size:.7rem;" onclick="quickLabel(${t.tweet_id}, 'Sell', this)">Sell</button>
                    <button class="btn btn-warning btn-sm py-0 px-1" style="font-size:.7rem;" onclick="quickLabel(${t.tweet_id}, 'Hold', this)">Hold</button>
                    <button class="btn btn-secondary btn-sm py-0 px-1" style="font-size:.7rem;" onclick="quickLabel(${t.tweet_id}, 'No Opinion', this)">N/O</button>
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('tweetPanelBody').innerHTML = html;
    // Stash by tweet_id so quickLabel can retrieve full row
    window._tweetPanelTweets = Object.fromEntries(data.tweets.map(t => [t.tweet_id, t]));
}

async function quickLabel(tweetId, label, btn) {
    const t = (window._tweetPanelTweets || {})[tweetId];
    if (!t) return;
    const original = btn.parentElement.innerHTML;
    btn.parentElement.innerHTML = '<span class="text-secondary small">Saving…</span>';
    try {
        const res = await fetch('/api/labels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tweet_id:         t.tweet_id,
                tweet:            t.tweet,
                ticker:           t.ticker,
                trading_date:     t.trading_date,
                label,
                model:            getModel(),
                model_prediction: (t.predictions || {})[getModel()] || null,
            }),
        });
        if (!res.ok) { btn.parentElement.innerHTML = original; return; }
        btn.parentElement.innerHTML = `<span class="text-success small">✓ Labeled as ${label}</span>`;
    } catch (e) {
        btn.parentElement.innerHTML = original;
    }
}

function renderDayPanel(data) {
    document.getElementById('dpTicker').textContent =
        `${data.ticker} — ${data.company}`;
    document.getElementById('dpDate').textContent = data.date_formatted;

    const model   = getModel();
    const mdata   = data.models[model] || Object.values(data.models)[0];
    const drift   = data.drift[model]  || Object.values(data.drift)[0]  || {};
    const price   = data.price;

    const body = document.getElementById('dayPanelBody');
    body.innerHTML = `
        ${renderPriceCard(price)}
        ${renderGaugeSection(mdata, model)}
        ${renderSentimentBar(mdata)}
        ${renderRollingDrift(mdata, drift)}
        ${renderModelComparison(data)}
        <div class="dp-disclaimer">
            ⚠ For informational purposes only. Not financial advice. Correlation ≠ causation.
        </div>
    `;

    // Draw gauge canvas after DOM insert
    if (mdata) requestAnimationFrame(() => drawGauge(mdata));
}

function renderPriceCard(price) {
    if (!price) return `<div class="dp-section"><div class="dp-section-title">Price</div><div class="text-secondary small">Price data unavailable for this date.</div></div>`;

    const ret     = price.daily_return;
    const retStr  = ret != null ? (ret >= 0 ? `+${ret.toFixed(2)}%` : `${ret.toFixed(2)}%`) : '—';
    const retColor= ret == null ? '#8b949e' : ret >= 0 ? COLORS.buy : COLORS.sell;
    const trend   = price.intraday_trend === 1
        ? '<span class="dp-badge dp-badge-green">▲ Bullish Day</span>'
        : '<span class="dp-badge dp-badge-red">▼ Bearish Day</span>';
    const vol     = price.volume != null ? (price.volume / 1e6).toFixed(1) + 'M shares' : '—';

    const fmt = v => v != null ? `$${Number(v).toFixed(2)}` : '—';

    return `
    <div class="dp-section">
        <div class="dp-section-title">Price</div>
        <div class="dp-price-hero">
            <span class="dp-close-price">${fmt(price.close)}</span>
            <span class="dp-return" style="color:${retColor}">${retStr}</span>
            ${trend}
        </div>
        <div class="dp-ohlc-grid">
            <div class="dp-ohlc-cell"><div class="dp-ohlc-label">Open</div><div class="dp-ohlc-val">${fmt(price.open)}</div></div>
            <div class="dp-ohlc-cell"><div class="dp-ohlc-label">High</div><div class="dp-ohlc-val" style="color:${COLORS.buy}">${fmt(price.high)}</div></div>
            <div class="dp-ohlc-cell"><div class="dp-ohlc-label">Low</div><div class="dp-ohlc-val" style="color:${COLORS.sell}">${fmt(price.low)}</div></div>
            <div class="dp-ohlc-cell"><div class="dp-ohlc-label">Volume</div><div class="dp-ohlc-val">${vol}</div></div>
        </div>
    </div>`;
}

function renderGaugeSection(mdata, model) {
    if (!mdata) return '';
    const score  = mdata.sentiment_score;
    const sig    = interpretSignalScore(score);
    const vol    = mdata.tweet_volume != null ? Math.round(mdata.tweet_volume) : 0;

    return `
    <div class="dp-section">
        <div class="dp-section-title">Sentiment — ${model.toUpperCase()}</div>
        <div class="dp-gauge-row">
            <div class="dp-gauge-wrap">
                <canvas id="gaugeCanvas" width="200" height="110"></canvas>
                <div class="dp-gauge-labels">
                    <span style="color:${COLORS.sell}">Bearish</span>
                    <span style="color:${COLORS.hold}">Neutral</span>
                    <span style="color:${COLORS.buy}">Bullish</span>
                </div>
            </div>
            <div class="dp-gauge-stats">
                <div class="dp-stat-score" style="color:${sig.color}">${score != null ? score.toFixed(3) : '—'}</div>
                <div class="dp-stat-label" style="color:${sig.color}">${sig.icon} ${sig.label}</div>
                <div class="dp-stat-sub">${vol} tweets</div>
                <div class="dp-dist-bars">
                    ${distBar('Buy',        mdata.buy_pct,        COLORS.buy)}
                    ${distBar('Sell',       mdata.sell_pct,       COLORS.sell)}
                    ${distBar('Hold',       mdata.hold_pct,       COLORS.hold)}
                    ${distBar('No Opinion', mdata.no_opinion_pct, COLORS.noOpinion)}
                </div>
            </div>
        </div>
    </div>`;
}

function distBar(label, pct, color) {
    const p = pct != null ? (pct * 100).toFixed(1) : 0;
    return `
    <div class="dp-dist-row">
        <span class="dp-dist-label">${label}</span>
        <div class="dp-dist-track">
            <div class="dp-dist-fill" style="width:${p}%;background:${color}"></div>
        </div>
        <span class="dp-dist-pct">${p}%</span>
    </div>`;
}

function renderSentimentBar(mdata) {
    if (!mdata || mdata.sentiment_score == null) return '';
    const score = mdata.sentiment_score;
    // Map score (-1..1) → position (0%..100%)
    const pct   = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));

    return `
    <div class="dp-section">
        <div class="dp-section-title">Sentiment Meter</div>
        <div class="dp-meter-wrap">
            <div class="dp-meter-track">
                <div class="dp-meter-needle" style="left:${pct}%"></div>
            </div>
            <div class="dp-meter-ends">
                <span style="color:${COLORS.sell}">◀ Bearish −1</span>
                <span class="dp-meter-val">${score.toFixed(3)}</span>
                <span style="color:${COLORS.buy}">+1 Bullish ▶</span>
            </div>
        </div>
    </div>`;
}

function renderRollingDrift(mdata, drift) {
    const r3 = mdata?.rolling_3day != null ? mdata.rolling_3day.toFixed(3) : '—';
    const r7 = mdata?.rolling_7day != null ? mdata.rolling_7day.toFixed(3) : '—';

    const flagHtml = drift && drift.any
        ? `<div class="dp-drift-flags">
            ${driftFlag('Distribution Drift', drift.drift_flag)}
            ${driftFlag('Volume Spike',        drift.volume_spike_flag)}
            ${driftFlag('Weak Signal',         drift.weak_signal_flag)}
            ${driftFlag('Price Divergence',    drift.divergence_flag)}
           </div>`
        : `<div class="dp-drift-ok">✓ No drift detected</div>`;

    return `
    <div class="dp-section">
        <div class="dp-section-row">
            <div class="dp-half">
                <div class="dp-section-title">Rolling Context</div>
                <div class="dp-kv"><span>3-Day Avg</span><span>${r3}</span></div>
                <div class="dp-kv"><span>7-Day Avg</span><span>${r7}</span></div>
            </div>
            <div class="dp-half">
                <div class="dp-section-title">Drift Status</div>
                ${flagHtml}
            </div>
        </div>
    </div>`;
}

function driftFlag(label, active) {
    return `<div class="dp-flag ${active ? 'dp-flag-on' : 'dp-flag-off'}">
        ${active ? '⚠' : '✓'} ${label}
    </div>`;
}

function renderModelComparison(data) {
    const models = Object.keys(data.models);
    if (models.length < 2) return '';

    const cells = models.map(m => {
        const d   = data.models[m];
        const s   = d.sentiment_score;
        const sig = interpretSignalScore(s);
        return `<div class="dp-model-cell">
            <div class="dp-model-name">${m.toUpperCase()}</div>
            <div class="dp-model-score" style="color:${sig.color}">${s != null ? s.toFixed(3) : '—'}</div>
            <div class="dp-model-label" style="color:${sig.color}">${sig.label}</div>
        </div>`;
    }).join('');

    return `
    <div class="dp-section">
        <div class="dp-section-title">Model Comparison</div>
        <div class="dp-model-grid">${cells}</div>
    </div>`;
}

// ── Gauge canvas (half-doughnut drawn via Canvas API) ──────────────

function drawGauge(mdata) {
    const canvas = document.getElementById('gaugeCanvas');
    if (!canvas) return;
    // Explicitly set pixel dimensions so CSS scaling never distorts the drawing
    canvas.width  = 200;
    canvas.height = 110;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H - 10, r = 80, rInner = 52;

    // Segments: sell → hold → buy (left to right = π to 0)
    const buy  = mdata.buy_pct        || 0;
    const hold = mdata.hold_pct       || 0;
    const sell = mdata.sell_pct       || 0;
    const nop  = mdata.no_opinion_pct || 0;

    const total = buy + hold + sell + nop || 1;
    const segs  = [
        { pct: sell / total, color: COLORS.sell },
        { pct: nop  / total, color: COLORS.noOpinion },
        { pct: hold / total, color: COLORS.hold },
        { pct: buy  / total, color: COLORS.buy  },
    ];

    let angle = Math.PI;
    segs.forEach(seg => {
        const span = seg.pct * Math.PI;
        ctx.beginPath();
        ctx.moveTo(cx + rInner * Math.cos(angle), cy + rInner * Math.sin(angle));
        ctx.arc(cx, cy, r,      angle, angle + span);
        ctx.arc(cx, cy, rInner, angle + span, angle, true);
        ctx.closePath();
        ctx.fillStyle = seg.color;
        ctx.fill();
        angle += span;
    });

    // Needle
    const score = mdata.sentiment_score || 0;
    const needleAngle = Math.PI + (score + 1) / 2 * Math.PI;
    const nx = cx + (rInner - 4) * Math.cos(needleAngle);
    const ny = cy + (rInner - 4) * Math.sin(needleAngle);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Center dot
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
}

// ── Signal interpretation ──────────────────────────────────────────

function interpretSignalScore(score) {
    if (score == null || isNaN(score)) return { label:'—', color:'#8b949e', icon:'→' };
    if (score >  0.3) return { label:'Strong Bullish', color:'#3fb950', icon:'↑↑' };
    if (score >  0.1) return { label:'Mild Bullish',   color:'#8adb88', icon:'↑'  };
    if (score < -0.3) return { label:'Strong Bearish', color:'#f85149', icon:'↓↓' };
    if (score < -0.1) return { label:'Mild Bearish',   color:'#ff9999', icon:'↓'  };
    return { label:'Neutral', color:'#e3b341', icon:'→' };
}

// kept for inline script compatibility
function interpretSignal(score) { return interpretSignalScore(score); }
