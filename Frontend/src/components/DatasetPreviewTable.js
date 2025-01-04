import React, { useState, useEffect } from "react";

const DatasetPreviewTable = ({ tableData, rowLimit, setRowLimit }) => {
  const columns = tableData.length > 0 ? Object.keys(tableData[0]) : [];

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center space-x-4 mb-4">
        <label htmlFor="row-limit" className="text-lg font-medium text-green-800">
          Number of Rows:
        </label>
        <select
          id="row-limit"
          value={rowLimit}
          onChange={(e) => setRowLimit(Number(e.target.value))}
          className="px-3 py-2 border rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-green-500"
        >
          <option value={5}>5</option>
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
          <option value={200}>200</option>
        </select>
      </div>

      <div
        className="overflow-x-auto bg-white rounded-lg shadow-md border border-gray-200"
        style={{ maxHeight: "500px", overflowY: "auto" }}
      >
        <table className="w-full text-left border-collapse">
          <thead className="bg-green-600 text-white sticky top-0 z-10">
            <tr>
              {columns.map((header) => (
                <th
                  key={header}
                  className="p-4 text-sm font-medium uppercase border-b"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableData.map((row, index) => (
              <tr key={index} className="hover:bg-gray-100 transition">
                {columns.map((col) => (
                  <td key={col} className="p-4 text-gray-700 text-sm border-b">
                    {row[col] !== null ? row[col].toString() : "N/A"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [tableData, setTableData] = useState([]);
  const [rowLimit, setRowLimit] = useState(10);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTableData = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `http://127.0.0.1:8000/get-data-head?row_limit=${rowLimit}`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch table data");
        }
        const data = await response.json();
        setTableData(data.head || []);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchTableData();
  }, [rowLimit]);

  return (
    <div>
      {loading ? (
        <div className="p-4 text-center text-green-600 font-medium">
          Loading data...
        </div>
      ) : error ? (
        <div className="p-4 text-center text-red-600 font-medium">
          Error: {error}
        </div>
      ) : (
        <DatasetPreviewTable
          tableData={tableData}
          rowLimit={rowLimit}
          setRowLimit={setRowLimit}
        />
      )}
    </div>
  );
};

export default Dashboard;
