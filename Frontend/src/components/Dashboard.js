import React, { useState, useEffect } from 'react';
import DataInfo from './Metrics';
import DistributionsOverview from './DistributionsOverview';
import ModelPerformanceChart from './ModelPerformanceChart';
import DatasetPreviewTable from './DatasetPreviewTable';

const Dashboard = () => {
  const [tableData, setTableData] = useState([]);
  const [rowLimit, setRowLimit] = useState(10);
  const [metricsData, setMetricsData] = useState(null);
  const [distributionsData, setDistributionsData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDataSequentially = async () => {
    try {
      setLoading(true);

      const tableResponse = await fetch(`http://127.0.0.1:8000/get-data-head?row_limit=${rowLimit}`);
      if (!tableResponse.ok) {
        throw new Error('Failed to fetch table data');
      }
      const tableResult = await tableResponse.json();
      setTableData(tableResult.head || []);

      const metricsResponse = await fetch('http://127.0.0.1:8000/get-data-summary/');
      if (!metricsResponse.ok) {
        throw new Error('Failed to fetch metrics data');
      }
      const metricsResult = await metricsResponse.json();
      setMetricsData(metricsResult);

      const distributionsResponse = await fetch('http://127.0.0.1:8000/distributions');
      if (!distributionsResponse.ok) {
        throw new Error('Failed to fetch distributions data');
      }
      const distributionsResult = await distributionsResponse.json();
      setDistributionsData(distributionsResult);

      const chartResponse = await fetch('http://127.0.0.1:8000/get-chart-data/');
      if (!chartResponse.ok) {
        throw new Error('Failed to fetch chart data');
      }
      const chartResult = await chartResponse.json();
      setChartData(chartResult);

      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDataSequentially();
  }, [rowLimit]);

  const columns = tableData.length > 0 ? Object.keys(tableData[0]) : [];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-green-600 mb-6">Dataset Preview</h2>

      {loading ? (
        <div className="p-4 text-center text-green-600 font-medium">Loading data...</div>
      ) : error ? (
        <div className="p-4 text-center text-red-600 font-medium">Error: {error}</div>
      ) : (
        <DatasetPreviewTable
          data={tableData}
          columns={columns}
          rowLimit={rowLimit}
          onRowLimitChange={setRowLimit}
        />
      )}

      <h2 className="text-2xl font-bold text-green-600 mb-6 mt-9">Data Insights</h2>
      <DataInfo data={metricsData} />

      <DistributionsOverview data={distributionsData} />

      <div className="mb-8 h-500">
        <ModelPerformanceChart data={chartData} />
      </div>
    </div>
  );
};

export default Dashboard;
