import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const ModelPerformanceChart = () => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/get-chart-data/');
        if (!response.ok) {
          throw new Error('Failed to fetch chart data');
        }
        const data = await response.json();
        setChartData(data || {});
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchChartData();
  }, []);

  if (loading) {
    return <p className="text-center text-blue-600">Loading Model Performance...</p>;
  }

  if (error) {
    return <p className="text-center text-red-600">Error: {error}</p>;
  }

  const barChartData = {
    labels: chartData.models,
    datasets: [
      {
        label: 'F1 Score',
        data: chartData.f1_scores,
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'], 
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: (tooltipItem) => `F1 Score: ${tooltipItem.raw}`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Models',
          color: '#333',
          font: {
            size: 14,
            weight: 'bold',
          },
        },
        ticks: {
          maxRotation: 0,
          minRotation: 0,
        },
        grid: {
          display: false, 
        },
      },
      y: {
        title: {
          display: true,
          text: 'F1 Score',
          color: '#333',
          font: {
            size: 14,
            weight: 'bold',
          },
        },
        beginAtZero: true,
        max: 1.0, 
        grid: {
          color: '#f0f0f0',
        },
      },
    },
    elements: {
      bar: {
        barThickness: 5,
        maxBarThickness: 6, 
        categoryPercentage: 0.6, 
        barPercentage: 0.9, 
      },
    },
    layout: {
      padding: {
        top: 10,
        bottom: 10,
      },
    },
  };

  return (
    <div className="bg-white p-6 shadow-md rounded-lg">
      <h2 className="text-2xl font-bold text-green-600 mb-6">Model Performance</h2>
      <div style={{ height: '600px' }}>
        <Bar data={barChartData} options={barChartOptions} />
      </div>
    </div>
  );
};

export default ModelPerformanceChart;
