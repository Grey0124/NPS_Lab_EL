import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: '🏠' },
    { path: '/monitoring', label: 'Monitoring', icon: '🔍' },
    { path: '/statistics', label: 'Statistics', icon: '📊' },
    { path: '/configuration', label: 'Configuration', icon: '⚙️' },
    { path: '/alerts', label: 'Alerts', icon: '🚨' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-transparent py-4">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center">
          {/* Logo and Title */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-sm font-bold">ARP</span>
              </div>
              <span className="text-black ml-2 font-bold text-xl">
                ARP Guardian
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-black hover:text-blue-600 font-medium transition-colors ${
                  isActive(item.path) ? 'text-blue-600' : ''
                }`}
              >
                <span className="mr-1">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="focus:outline-none"
            >
              <svg className="w-6 h-6 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {isMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden mt-4 bg-white shadow-lg rounded-lg p-4">
            <div className="flex flex-col space-y-3">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsMenuOpen(false)}
                  className={`text-black hover:text-blue-600 font-medium p-2 rounded ${
                    isActive(item.path) ? 'bg-blue-50 text-blue-600' : ''
                  }`}
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.label}
                </Link>
              ))}
              <button
                onClick={() => setIsMenuOpen(false)}
                className="text-red-600 hover:text-red-800 font-medium p-2 text-left"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 