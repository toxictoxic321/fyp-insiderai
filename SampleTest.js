import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const SampleTest = () => {
    const [keyword, setKeyword] = useState("");
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleTest = async () => {
        setLoading(true);
        try {
            const response = await fetch(`http://127.0.0.1:8000/api/predict/${keyword}`);
            const result = await response.json();
            setData(result);
        } catch (error) {
            alert("Backend Connection Failed!");
        }
        setLoading(false);
    };

    // Prepare data for the Chart
    const chartData = data ? {
        labels: ['-5mo', '-4mo', '-3mo', '-2mo', '-1mo', 'Today', '+1mo', '+2mo', '+3mo', '+4mo', '+5mo', '+6mo'],
        datasets: [
            {
                label: 'Historical & Predicted Trend',
                data: [
                    ...data.historical.map(h => h.search_volume),
                    ...data.forecast.map(f => f.value)
                ],
                borderColor: '#3267E3',
                backgroundColor: 'rgba(50, 103, 227, 0.2)',
                fill: true,
                tension: 0.4, // Makes the line curvy/smooth
            }
        ]
    } : null;

    return (
        <div style={{ padding: '40px', maxWidth: '800px', margin: '0 auto', fontFamily: 'Inter' }}>
            <h2 style={{ color: '#1a1a1a' }}>📊 TrendPredictor™ Lab</h2>
            <div style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
                <input 
                    type="text" 
                    className="form-control"
                    placeholder="Enter Keyword (e.g. Mushroom Coffee)" 
                    style={{ padding: '10px', flex: 1, borderRadius: '8px', border: '1px solid #ddd' }}
                    onChange={(e) => setKeyword(e.target.value)} 
                />
                <button 
                    onClick={handleTest} 
                    style={{ padding: '10px 20px', backgroundColor: '#3267E3', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer' }}
                >
                    {loading ? "Analyzing..." : "Run AI Model"}
                </button>
            </div>

            {data && (
                <div style={{ background: 'white', padding: '20px', borderRadius: '15px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                    <h4 style={{ marginBottom: '20px' }}>Analysis for: <span style={{ color: '#3267E3' }}>{data.keyword}</span></h4>
                    <div style={{ height: '400px' }}>
                        <Line data={chartData} options={{ maintainAspectRatio: false }} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default SampleTest;