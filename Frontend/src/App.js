import React from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header />
      <main className="flex-grow p-6">
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
