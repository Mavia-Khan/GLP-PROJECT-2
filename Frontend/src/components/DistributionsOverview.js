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
ChartJS.defaults.datasets.bar.maxBarThickness = 150;

const DistributionsOverview = () => {
  const [distributions, setDistributions] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDistributions = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/distributions');
        if (!response.ok) {
          throw new Error('Failed to fetch distributions data');
        }
        const data = await response.json();
        setDistributions(data || {});
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchDistributions();
  }, []);

  if (loading) {
    return <p className="text-center text-blue-600">Loading Distributions...</p>;
  }

  if (error) {
    return <p className="text-center text-red-600">Error: {error}</p>;
  }

  const createBarChartData = (data, label) => ({
    labels: Object.keys(data || {}),
    datasets: [
      {
        label,
        data: Object.values(data || {}),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
        ],
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
      },
    ],
  });

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (tooltipItem) {
            const fullLabel = Object.keys(
              distributions[tooltipItem.dataset.label] || {}
            )[tooltipItem.dataIndex];
            return `${fullLabel}: ${tooltipItem.raw}`;
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Categories',
          color: '#333',
          font: {
            size: 14,
            weight: 'bold',
          },
        },
        ticks: {
          callback: function (val, index) {
            const fullLabel = this.getLabelForValue(val);
            return fullLabel.length > 10 ? `${fullLabel.slice(0, 10)}...` : fullLabel;
          },
          maxRotation: 45,
          minRotation: 0,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Count',
          color: '#333',
          font: {
            size: 14,
            weight: 'bold',
          },
        },
        beginAtZero: true,
        grid: {
          color: '#f0f0f0',
        },
      },
    },
  };

  const renderChart = (key, title, color) => {
    const data = distributions[key];
    if (!data) return null;
    return (
      <div className="bg-gray-50 p-4 shadow-md rounded-lg">
        <h3
          className="text-xl font-semibold mb-4"
          style={{ color: color }} 
        >
          {title}
        </h3>
        <Bar
          data={createBarChartData(data, title)}
          options={{
            ...chartOptions,
            plugins: {
              ...chartOptions.plugins,
              tooltip: {
                ...chartOptions.plugins.tooltip,
                callbacks: {
                  ...chartOptions.plugins.tooltip.callbacks,
                  title: function (tooltipItems) {
                    const index = tooltipItems[0].dataIndex;
                    return Object.keys(data)[index]; 
                  },
                },
              },
            },
          }}
        />
      </div>
    );
  };
  
  return (
    <div className="mt-6 bg-white p-6 shadow-md rounded-lg">
      <h2 className="text-2xl font-bold text-green-600 mb-6">Distributions Overview</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {renderChart('total_employed', 'Total Employed', '#FF5733')} 
        {renderChart('income_above_limit', 'Income Above Limit', '#33B5FF')} 
        {renderChart('citizenship', 'Citizenship', '#E91E63')} 
        {renderChart('education', 'Education', '#FFC107')} 
      </div>
    </div>
  );
  
};

export default DistributionsOverview;
