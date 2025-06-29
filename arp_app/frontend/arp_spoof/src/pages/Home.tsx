import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  const features = [
    {
      icon: '🔍',
      title: 'Real-time Monitoring',
      description: 'Continuously monitor your network for ARP spoofing attacks with instant detection capabilities.',
      color: 'from-blue-400 to-purple-500'
    },
    {
      icon: '📊',
      title: 'Advanced Analytics',
      description: 'Comprehensive statistics and detailed reports on network security threats and patterns.',
      color: 'from-purple-500 to-blue-400'
    },
    {
      icon: '🚨',
      title: 'Smart Alerts',
      description: 'Get notified immediately when suspicious activities are detected with customizable alert settings.',
      color: 'from-pink-500 to-purple-500'
    },
    {
      icon: '⚙️',
      title: 'Easy Configuration',
      description: 'Simple and intuitive interface to configure detection parameters and system settings.',
      color: 'from-blue-500 to-purple-400'
    }
  ];

  const stats = [
    { label: 'Detection Rate', value: '99.9%', description: 'High accuracy in threat detection' },
    { label: 'Response Time', value: '< 1s', description: 'Instant threat response' },
    { label: 'False Positives', value: '< 0.1%', description: 'Minimal false alarms' },
    { label: 'Network Coverage', value: '100%', description: 'Complete network protection' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200">
      <div className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h1 className="text-5xl md:text-6xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow-md mb-2">ARP Guardian</h1>
          <p className="text-lg text-gray-600 mb-4">Advanced ARP Spoofing Detection System</p>
          <p className="text-gray-500 mb-6">Protect your network from man-in-the-middle attacks with our intelligent ARP spoofing detection technology. Monitor, detect, and respond to threats in real-time.</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/monitoring" className="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-3 px-8 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 scale-105 transition-all">Start Monitoring</Link>
            <Link to="/configuration" className="border-2 border-blue-600 text-blue-600 hover:bg-blue-50 font-semibold py-3 px-8 rounded-lg transition-colors">Configure System</Link>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <div key={index} className="bg-white/80 border border-purple-200 rounded-xl shadow p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">{stat.value}</div>
              <div className="text-sm font-semibold text-gray-700 mb-1">{stat.label}</div>
              <div className="text-xs text-gray-500">{stat.description}</div>
            </div>
          ))}
        </div>

        {/* Features Section */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <h2 className="text-3xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-4">Key Features</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="bg-gradient-to-br from-white via-blue-50 to-purple-50 rounded-lg p-6 shadow hover:shadow-lg transition-shadow flex flex-col items-center">
                <div className={`w-14 h-14 bg-gradient-to-r ${feature.color} text-white rounded-full flex items-center justify-center text-3xl mb-4 shadow-lg`}>{feature.icon}</div>
                <h3 className="text-lg font-semibold text-blue-700 mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm text-center">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* How It Works Section */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h2 className="text-3xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-6 text-center">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto shadow-lg">1</div>
              <h3 className="text-xl font-semibold text-blue-700">Monitor</h3>
              <p className="text-gray-600">Continuously monitor network traffic and ARP tables for suspicious activities and inconsistencies.</p>
            </div>
            <div className="text-center space-y-4">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto shadow-lg">2</div>
              <h3 className="text-xl font-semibold text-purple-700">Detect</h3>
              <p className="text-gray-600">Use advanced algorithms to identify ARP spoofing attempts and potential security threats.</p>
            </div>
            <div className="text-center space-y-4">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-green-400 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto shadow-lg">3</div>
              <h3 className="text-xl font-semibold text-green-700">Respond</h3>
              <p className="text-gray-600">Immediately alert administrators and take automated actions to mitigate security risks.</p>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-8 text-white shadow-xl">
          <h2 className="text-3xl font-bold mb-4 drop-shadow">Ready to Secure Your Network?</h2>
          <p className="text-xl mb-6 opacity-90">Start monitoring your network for ARP spoofing attacks today.</p>
          <Link to="/monitoring" className="inline-block bg-white text-blue-600 font-semibold py-3 px-8 rounded-lg hover:bg-blue-50 hover:text-purple-700 transition-colors shadow">Get Started Now</Link>
        </div>
      </div>
    </div>
  );
};

export default Home; 