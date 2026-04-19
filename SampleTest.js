import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS, CategoryScale, LinearScale,
    PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// ── Countries ─────────────────────────────────────────────────────────────────
const COUNTRIES = [
    { code: "worldwide", label: "🌍 Worldwide" },
    { code: "US", label: "🇺🇸 United States" },
    { code: "GB", label: "🇬🇧 United Kingdom" },
    { code: "PK", label: "🇵🇰 Pakistan" },
    { code: "IN", label: "🇮🇳 India" },
    { code: "CA", label: "🇨🇦 Canada" },
    { code: "AU", label: "🇦🇺 Australia" },
    { code: "DE", label: "🇩🇪 Germany" },
    { code: "FR", label: "🇫🇷 France" },
    { code: "BR", label: "🇧🇷 Brazil" },
    { code: "MX", label: "🇲🇽 Mexico" },
    { code: "SA", label: "🇸🇦 Saudi Arabia" },
    { code: "AE", label: "🇦🇪 UAE" },
    { code: "NG", label: "🇳🇬 Nigeria" },
    { code: "ZA", label: "🇿🇦 South Africa" },
    { code: "JP", label: "🇯🇵 Japan" },
    { code: "KR", label: "🇰🇷 South Korea" },
    { code: "ID", label: "🇮🇩 Indonesia" },
    { code: "PH", label: "🇵🇭 Philippines" },
    { code: "TR", label: "🇹🇷 Turkey" },
    { code: "EG", label: "🇪🇬 Egypt" },
    { code: "IT", label: "🇮🇹 Italy" },
    { code: "ES", label: "🇪🇸 Spain" },
    { code: "NL", label: "🇳🇱 Netherlands" },
    { code: "SE", label: "🇸🇪 Sweden" },
    { code: "AR", label: "🇦🇷 Argentina" },
    { code: "CO", label: "🇨🇴 Colombia" },
    { code: "BD", label: "🇧🇩 Bangladesh" },
    { code: "MY", label: "🇲🇾 Malaysia" },
    { code: "TH", label: "🇹🇭 Thailand" },
    { code: "VN", label: "🇻🇳 Vietnam" },
];

// ── Chart divider plugin ──────────────────────────────────────────────────────
const dividerPlugin = {
    id: 'forecastDivider',
    afterDraw(chart) {
        const { x: xScale, y: yScale } = chart.scales;
        if (!xScale || !yScale) return;
        const xPos = (xScale.getPixelForValue(23) + xScale.getPixelForValue(24)) / 2;
        const ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = 'rgba(220,53,69,0.6)';
        ctx.lineWidth = 1.8;
        ctx.moveTo(xPos, yScale.top);
        ctx.lineTo(xPos, yScale.bottom);
        ctx.stroke();
        ctx.font = '11px sans-serif';
        ctx.fillStyle = 'rgba(220,53,69,0.75)';
        ctx.fillText('Forecast ▶', xPos + 5, yScale.top + 14);
        ctx.restore();
    }
};
ChartJS.register(dividerPlugin);

// ── Badge config ──────────────────────────────────────────────────────────────
const BADGE = {
    seasonal:      { color: '#f0a500', label: '📅 Seasonal' },
    trending_up:   { color: '#28a745', label: '📈 Trending Up' },
    trending_down: { color: '#dc3545', label: '📉 Trending Down' },
    stable:        { color: '#6c757d', label: '➡️ Stable' },
};

const fmtDate = (s) => {
    const d = new Date(s);
    return `${d.toLocaleString('default', { month: 'short' })} '${String(d.getFullYear()).slice(2)}`;
};

// ── Component ─────────────────────────────────────────────────────────────────
export default function SampleTest() {
    const [keyword, setKeyword]   = useState('');
    const [geo, setGeo]           = useState('worldwide');
    const [data, setData]         = useState(null);
    const [loading, setLoading]   = useState(false);
    const [error, setError]       = useState(null);
    const [fallback, setFallback] = useState(false);

    const analyze = async () => {
        if (!keyword.trim()) return;
        setLoading(true); setData(null); setError(null); setFallback(false);
        try {
            const geoParam = geo === 'worldwide' ? '' : geo;
            const res = await fetch(
                `http://127.0.0.1:8000/api/predict/${encodeURIComponent(keyword.trim())}?geo=${geoParam}`
            );
            const r = await res.json();
            if (r.status === 'error')       setError(r.message);
            else if (r.status === 'low_quality') setError(`⚠️ ${r.message}`);
            else {
                if (r.geo_fallback) setFallback(true);
                setData(r);
            }
        } catch {
            setError('❌ Backend is down!\n\ncd backend\n.\\venv\\Scripts\\activate\nuvicorn main:app --reload');
        }
        setLoading(false);
    };

    // ── Chart ─────────────────────────────────────────────────────────────────
    const histLabels = data ? data.historical.map(h => fmtDate(h.date)) : [];
    const allLabels  = data ? [...histLabels, '+1', '+2', '+3', '+4', '+5', '+6'] : [];

    const chartData = data ? {
        labels: allLabels,
        datasets: [
            {
                label: 'Historical (Google Trends)',
                data: [...data.historical.map(h => h.search_volume), ...Array(6).fill(null)],
                borderColor: '#3267E3',
                backgroundColor: 'rgba(50,103,227,0.07)',
                fill: true, tension: 0.35, pointRadius: 3, pointHoverRadius: 6, borderWidth: 2,
            },
            {
                label: '6-Month Forecast',
                data: [...Array(24).fill(null), ...data.forecast.map(f => f.value)],
                borderColor: '#ff6b35',
                backgroundColor: 'rgba(255,107,53,0.10)',
                fill: true, tension: 0.35, pointRadius: 4, pointHoverRadius: 7, borderWidth: 2.5,
            }
        ]
    } : null;

    const chartOpts = {
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: true, position: 'top', labels: { font: { size: 12 }, padding: 14 } },
            tooltip: { callbacks: { label: ctx => ctx.parsed.y == null ? null : `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}` } }
        },
        scales: {
            x: { ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 }, maxTicksLimit: 16 }, grid: { color: 'rgba(0,0,0,0.04)' } },
            y: {
                beginAtZero: false, grid: { color: 'rgba(0,0,0,0.04)' },
                ticks: { font: { size: 11 }, callback: v => Math.round(v) },
                title: { display: true, text: 'Google Trends Interest (0–100)', font: { size: 11 }, color: '#999' }
            }
        }
    };

    // ── Stats ─────────────────────────────────────────────────────────────────
    const avgHist  = data ? Math.round(data.historical.reduce((a, b) => a + b.search_volume, 0) / data.historical.length) : 0;
    const peakHist = data ? Math.round(Math.max(...data.historical.map(h => h.search_volume))) : 0;
    const avgFcast = data ? Math.round(data.forecast.reduce((a, b) => a + b.value, 0) / data.forecast.length) : 0;
    const rising   = data ? data.forecast[5].value > data.historical[data.historical.length - 1].search_volume : false;
    const badge    = data ? (BADGE[data.product_type] || BADGE.stable) : null;

    // ── Styles ────────────────────────────────────────────────────────────────
    const card = { background: '#fff', padding: 28, borderRadius: 16, boxShadow: '0 4px 24px rgba(0,0,0,0.07)', border: '1px solid #f0f0f0' };
    const pill = (bg, fg = '#fff') => ({ background: bg, color: fg, padding: '5px 14px', borderRadius: 20, fontSize: 13, fontWeight: 600 });

    return (
        <div style={{ padding: '36px 20px', maxWidth: 1100, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }}>

            {/* Header */}
            <h2 style={{ fontWeight: 700, fontSize: 24, color: '#1a1a2e', marginBottom: 6 }}>📈 Product Trend Predictor</h2>
            <p style={{ color: '#777', fontSize: 13, marginBottom: 26 }}>
                Enter any product → select a country → get 24-month history + 6-month XGBoost forecast.
                If your country has no data, Worldwide is used automatically.
            </p>

            {/* Controls */}
            <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
                <input
                    style={{ flex: '1 1 280px', padding: '12px 16px', borderRadius: 10, border: '1.5px solid #ddd', fontSize: 15, outline: 'none' }}
                    placeholder="e.g. football, sunscreen, red lipstick, baby lotion..."
                    value={keyword}
                    onChange={e => setKeyword(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && analyze()}
                    onFocus={e => e.target.style.border = '1.5px solid #3267E3'}
                    onBlur={e  => e.target.style.border = '1.5px solid #ddd'}
                />
                <select
                    value={geo} onChange={e => setGeo(e.target.value)}
                    style={{ flex: '0 1 200px', padding: '12px', borderRadius: 10, border: '1.5px solid #ddd', fontSize: 14, background: '#fff', outline: 'none' }}
                >
                    {COUNTRIES.map(c => <option key={c.code} value={c.code}>{c.label}</option>)}
                </select>
                <button
                    onClick={analyze} disabled={loading}
                    style={{ flex: '0 0 auto', padding: '12px 26px', background: loading ? '#aab4c8' : '#3267E3', color: '#fff', border: 'none', borderRadius: 10, fontWeight: 600, fontSize: 15, cursor: loading ? 'not-allowed' : 'pointer' }}
                >
                    {loading ? '⏳ Analyzing...' : 'Analyze Trend'}
                </button>
            </div>

            {/* Geo fallback notice */}
            {fallback && data && (
                <div style={{ background: '#fff8e1', border: '1px solid #ffc107', borderRadius: 8, padding: '10px 16px', marginBottom: 14, fontSize: 13, color: '#664d03' }}>
                    📍 No regional data found for your selected country — showing <strong>Worldwide</strong> data instead.
                </div>
            )}

            {/* Error */}
            {error && (
                <div style={{ background: '#fff0f0', border: '1px solid #f5c6cb', borderRadius: 10, padding: '14px 18px', marginBottom: 18, fontSize: 14, whiteSpace: 'pre-wrap', color: '#721c24' }}>
                    {error}
                </div>
            )}

            {/* Loading */}
            {loading && (
                <div style={{ textAlign: 'center', padding: '60px 20px', background: '#f8f9ff', borderRadius: 16, border: '1px dashed #c5d0f0', color: '#3267E3' }}>
                    <div style={{ fontSize: 32, marginBottom: 10 }}>🔄</div>
                    <strong>Fetching from Google Trends via SerpApi...</strong>
                    <div style={{ color: '#999', marginTop: 8, fontSize: 13 }}>Usually takes 3–6 seconds. Running XGBoost model after.</div>
                </div>
            )}

            {/* Results */}
            {data && !loading && (
                <div style={card}>
                    {/* Badges */}
                    <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 18, flexWrap: 'wrap' }}>
                        <span style={pill(badge.color)}>{badge.label}</span>
                        <span style={pill('#eef2ff', '#3267E3')}>📍 {data.geo}</span>
                        <span style={{ marginLeft: 'auto', color: '#bbb', fontSize: 12 }}>24-month history + 6-month forecast</span>
                    </div>

                    {/* Chart */}
                    <div style={{ height: 420 }}>
                        <Line data={chartData} options={chartOpts} />
                    </div>

                    {/* Disclaimer */}
                    <div style={{ marginTop: 12, padding: '10px 14px', background: '#f8f9fc', borderRadius: 8, fontSize: 12, color: '#888', borderLeft: '3px solid #3267E3' }}>
                        💡 <strong>Note:</strong> Google Trends shows <em>relative</em> search interest (0–100), not sales volume.
                        The trend pattern (peaks, troughs, direction) reflects real marketplace demand.
                    </div>

                    {/* Stats */}
                    <div style={{ display: 'flex', gap: 14, marginTop: 20, flexWrap: 'wrap' }}>
                        {[
                            { label: 'Avg Interest (24mo)', val: avgHist,  color: '#3267E3' },
                            { label: 'Peak Interest',        val: peakHist, color: '#3267E3' },
                            { label: '6-Month Outlook',     val: rising ? '▲ Rising' : '▼ Falling', color: rising ? '#28a745' : '#dc3545' },
                            { label: 'Avg Forecast',        val: avgFcast, color: '#ff6b35' },
                        ].map((s, i) => (
                            <div key={i} style={{ flex: '1 1 130px', textAlign: 'center', background: '#f8f9fc', padding: '14px 8px', borderRadius: 12, border: '1px solid #eee' }}>
                                <div style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.val}</div>
                                <div style={{ fontSize: 12, color: '#999', marginTop: 4 }}>{s.label}</div>
                            </div>
                        ))}
                    </div>

                    {/* Monthly breakdown */}
                    <div style={{ marginTop: 20 }}>
                        <h4 style={{ fontSize: 13, color: '#666', marginBottom: 10, fontWeight: 600 }}>📊 Monthly Forecast</h4>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                            {data.forecast.map((f, i) => (
                                <div key={i} style={{ flex: '1 1 70px', background: '#fff7f4', border: '1px solid #ffd6c2', borderRadius: 10, padding: '10px 6px', textAlign: 'center' }}>
                                    <div style={{ fontSize: 11, color: '#aaa' }}>+{f.month} mo</div>
                                    <div style={{ fontSize: 18, fontWeight: 700, color: '#ff6b35' }}>{Math.round(f.value)}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}