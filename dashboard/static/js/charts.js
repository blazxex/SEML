// Chart.js helpers for the stock sentiment dashboard

let sentimentChart = null;
let distributionChart = null;

const CHART_COLORS = {
    buy: "rgba(72,199,142,0.85)",
    sell: "rgba(255,99,132,0.85)",
    hold: "rgba(255,206,86,0.85)",
    noOpinion: "rgba(150,150,200,0.6)",
    sentiment: "rgba(100,149,237,1)",
    rolling3: "rgba(100,220,237,1)",
    rolling7: "rgba(180,100,237,1)",
    price: "rgba(255,165,0,1)",
};

const SMOOTHING_LABELS = {
    daily: "Daily (raw)",
    "3day": "3-Day Rolling Avg",
    "7day": "7-Day Rolling Avg",
};

function buildSentimentChart(labels, sentimentData, rolling3Data, rolling7Data, priceData, smoothing) {
    const ctx = document.getElementById("sentimentChart").getContext("2d");
    if (sentimentChart) sentimentChart.destroy();

    // Pick which series to show based on smoothing selector
    const sentDataset = smoothing === "3day"
        ? { label: "3-Day Avg Sentiment", data: rolling3Data, borderColor: CHART_COLORS.rolling3, backgroundColor: "rgba(100,220,237,0.1)" }
        : smoothing === "7day"
        ? { label: "7-Day Avg Sentiment", data: rolling7Data, borderColor: CHART_COLORS.rolling7, backgroundColor: "rgba(180,100,237,0.1)" }
        : { label: "Sentiment Score (Daily)", data: sentimentData, borderColor: CHART_COLORS.sentiment, backgroundColor: "rgba(100,149,237,0.1)" };

    sentimentChart = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    ...sentDataset,
                    yAxisID: "ySentiment",
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: 2,
                },
                {
                    label: "Close Price ($)",
                    data: priceData,
                    borderColor: CHART_COLORS.price,
                    backgroundColor: "rgba(255,165,0,0.08)",
                    yAxisID: "yPrice",
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: { legend: { labels: { color: "#c0c0e0" } } },
            scales: {
                x: { ticks: { color: "#8080a0", maxTicksLimit: 10 }, grid: { color: "#2a2a4a" } },
                ySentiment: {
                    type: "linear",
                    position: "left",
                    ticks: { color: sentDataset.borderColor },
                    grid: { color: "#2a2a4a" },
                    title: { display: true, text: "Sentiment Score", color: sentDataset.borderColor },
                },
                yPrice: {
                    type: "linear",
                    position: "right",
                    ticks: { color: CHART_COLORS.price },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: "Close Price ($)", color: CHART_COLORS.price },
                },
            },
        },
    });

    // Update card header title
    document.getElementById("sentimentChartTitle").textContent =
        `Sentiment Score (${SMOOTHING_LABELS[smoothing]}) vs. Close Price`;
}

function buildDistributionChart(labels, buyPct, holdPct, sellPct, noPct) {
    const ctx = document.getElementById("distributionChart").getContext("2d");
    if (distributionChart) distributionChart.destroy();

    distributionChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                { label: "Buy",        data: buyPct.map(v => +(v * 100).toFixed(1)),  backgroundColor: CHART_COLORS.buy,       stack: "stack" },
                { label: "Hold",       data: holdPct.map(v => +(v * 100).toFixed(1)), backgroundColor: CHART_COLORS.hold,      stack: "stack" },
                { label: "Sell",       data: sellPct.map(v => +(v * 100).toFixed(1)), backgroundColor: CHART_COLORS.sell,      stack: "stack" },
                { label: "No Opinion", data: noPct.map(v => +(v * 100).toFixed(1)),   backgroundColor: CHART_COLORS.noOpinion, stack: "stack" },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#c0c0e0" } } },
            scales: {
                x: { stacked: true, ticks: { color: "#8080a0", maxTicksLimit: 10 }, grid: { color: "#2a2a4a" } },
                y: { stacked: true, ticks: { color: "#8080a0" }, grid: { color: "#2a2a4a" }, max: 100,
                     title: { display: true, text: "%", color: "#8080a0" } },
            },
        },
    });
}

async function loadData() {
    const ticker   = document.getElementById("tickerSelect").value;
    const model    = document.getElementById("modelSelect").value;
    const smoothing = document.getElementById("smoothingSelect").value;

    const sentResp = await fetch(`/api/sentiment?ticker=${ticker}&model=${model}`);
    if (!sentResp.ok) {
        document.getElementById("errorBanner").textContent = await sentResp.text();
        document.getElementById("errorBanner").style.display = "block";
        return;
    }
    document.getElementById("errorBanner").style.display = "none";
    const sent = await sentResp.json();

    buildSentimentChart(
        sent.dates,
        sent.sentiment_score,
        sent.rolling_3day_sentiment,
        sent.rolling_7day_sentiment,
        sent.close_price,
        smoothing
    );
    buildDistributionChart(sent.dates, sent.buy_pct, sent.hold_pct, sent.sell_pct, sent.no_opinion_pct);

    // Drift
    const driftResp = await fetch(`/api/drift?ticker=${ticker}&model=${model}`);
    if (driftResp.ok) {
        const drift = await driftResp.json();
        const banner = document.getElementById("driftBanner");
        if (drift.any_active) {
            banner.classList.add("active");
            const flags = [];
            if (drift.drift_flag.some(Boolean))        flags.push("Label Distribution Drift");
            if (drift.volume_spike_flag.some(Boolean)) flags.push("Volume Spike");
            if (drift.weak_signal_flag.some(Boolean))  flags.push("Weak Signal (High No Opinion)");
            if (drift.divergence_flag.some(Boolean))   flags.push("Sentiment↔Price Divergence");
            banner.innerHTML = `<strong>⚠ Data Drift Detected:</strong> ${flags.join(" | ")}`;
        } else {
            banner.classList.remove("active");
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("tickerSelect").addEventListener("change", loadData);
    document.getElementById("modelSelect").addEventListener("change", loadData);
    document.getElementById("smoothingSelect").addEventListener("change", loadData);
    loadData();
});
