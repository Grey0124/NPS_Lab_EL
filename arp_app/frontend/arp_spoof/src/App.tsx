import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Monitoring from './pages/Monitoring';
import Statistics from './pages/Statistics';
import Configuration from './pages/Configuration';
import Alerts from './pages/Alerts';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-blue-100">
          <ProtectedRoute>
            <Navbar />
            <main className="container mx-auto px-4 py-8">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/monitoring" element={<Monitoring />} />
                <Route path="/statistics" element={<Statistics />} />
                <Route path="/configuration" element={<Configuration />} />
                <Route path="/alerts" element={<Alerts />} />
              </Routes>
            </main>
          </ProtectedRoute>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
