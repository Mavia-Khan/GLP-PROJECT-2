import React, { useState, useEffect } from 'react';

const Metrics = ({ data }) => (
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
    <div className="p-6 bg-white shadow-md rounded-lg border">
    <h3 className="text-lg font-semibold mb-4" style={{ color: '#33B5FF' }}>General Information</h3>
    <p className="text-sm text-gray-600"><strong>Total Rows:</strong> {data.general_info.total_rows}</p>
      <p className="text-sm text-gray-600"><strong>Total Columns:</strong> {data.general_info.total_columns}</p>
      <p className="text-sm text-gray-600"><strong>Numerical Columns:</strong> {data.general_info.numerical_columns}</p>
      <p className="text-sm text-gray-600"><strong>Categorical Columns:</strong> {data.general_info.categorical_columns}</p>
    </div>

    <div className="p-6 bg-white shadow-md rounded-lg border">
      <h3 className="text-lg font-semibold text-green-600 mb-4">Data Completeness</h3>
      <p className="text-sm text-gray-600">
        <strong>Total Missing Values:</strong> {data.data_completeness.total_missing_values}
      </p>
      <p className="text-sm text-gray-600">
        <strong>Percentage Missing:</strong> {data.data_completeness.percentage_missing}%
      </p>
      <p className="text-sm text-gray-600">
        <strong>Rows with Missing Values:</strong> {data.data_completeness.rows_with_missing_values}
      </p>
      <div className="mt-4">
        <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="absolute h-full bg-green-500"
            style={{ width: `${100 - data.data_completeness.percentage_missing}%` }}
          />
        </div>
      </div>
    </div>

    <div className="p-6 bg-white shadow-md rounded-lg border">
      <h3 className="text-lg font-semibold text-purple-600 mb-4">Numerical Data Overview</h3>
      <p className="text-sm text-gray-600"><strong>Mean:</strong> {data.numerical_data.overall_mean}</p>
      <p className="text-sm text-gray-600"><strong>Std Dev:</strong> {data.numerical_data.overall_std}</p>
      <p className="text-sm text-gray-600"><strong>Min:</strong> {data.numerical_data.overall_min}</p>
      <p className="text-sm text-gray-600"><strong>Max:</strong> {data.numerical_data.overall_max}</p>
    </div>

    <div className="p-6 bg-white shadow-md rounded-lg border col-span-1 lg:col-span-3">
      <h3 className="text-lg font-semibold text-orange-600 mb-4">Categorical Data Overview</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(data.categorical_data.most_frequent_value).map(([key, value]) => (
          <div key={key} className="bg-gray-50 p-4 rounded-lg shadow-sm border">
            <p className="text-sm font-semibold text-gray-700 capitalize">{key}</p>
            <p className="text-sm text-gray-600">{value}</p>
          </div>
        ))}
        <div className="bg-gray-50 p-4 rounded-lg shadow-sm border">
          <p className="text-sm font-semibold text-gray-700">Total Unique Values</p>
          <p className="text-sm text-gray-600">{data.categorical_data.total_unique_values}</p>
        </div>
      </div>
    </div>
  </div>
);

const DataInfo = () => {
  const [dataSummary, setDataSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/get-data-summary/');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const data = await response.json();
        setDataSummary(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="p-6 bg-gray-50 rounded-lg shadow-md">
        <p className="text-lg font-semibold text-blue-600">Loading data summary...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 rounded-lg shadow-md">
        <p className="text-lg font-semibold text-red-600">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 rounded-lg shadow-md">
      <h2 className="text-2xl font-semibold text-blue-600 mb-6">Dataset Summary</h2>
      <Metrics data={dataSummary} />
    </div>
  );
};

export default DataInfo;
