import React from 'react';

const Header = () => (
  <header className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 shadow-md sticky top-0 z-10">
    <div className="flex items-center space-x-4">
      <h1 className="text-3xl font-bold">Income Inequality Analysis Dashboard</h1>
    </div>
    <p className="mt-2 text-lg">Explore and analyze income inequality data for meaningful insights</p>
  </header>
);

export default Header;
