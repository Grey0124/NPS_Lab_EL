import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import type { DetectionStats, DetectionRecord } from '../services/api';

const Statistics: React.FC = () => {
  const [stats, setStats] = useState<DetectionStats>({
    total_packets: 0,
    arp_packets: 0,
    detected_attacks: 0,
    monitoring_status: 'stopped',
    current_interface: null,
    recent_detections: []
  });
  const [recentDetections, setRecentDetections] = useState<DetectionRecord[]>([]);
  const [timeRange, setTimeRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStatistics();
  }, [timeRange]);

  const loadStatistics = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Load statistics from backend
      const statsData = await apiService.getStatistics();
      setStats(statsData);
      
      // Load recent detections
      const detectionsResponse = await apiService.getRecentDetections(10);
      setRecentDetections(detectionsResponse.detections);
      
    } catch (error) {
      console.error('Failed to load statistics:', error);
      setError('Failed to load statistics from the backend');
    } finally {
      setIsLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'resolved': return 'bg-green-100 text-green-800';
      case 'investigating': return 'bg-blue-100 text-blue-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="bg-white/80 border border-red-200 rounded-xl shadow-xl p-8 text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-red-700 mb-4">Error Loading Statistics</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={loadStatistics}
            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-2 px-6 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 transition-all"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-2">Detection Statistics</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">Comprehensive analytics and insights about ARP spoofing detection performance and network security metrics.</p>
        </div>

        {/* Time Range Selector */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8 flex items-center justify-between">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4 absolute left-0 right-0 top-0"></div>
          <h2 className="text-xl font-bold text-blue-700">Time Range</h2>
          <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
            <option value="1d">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">{stats.detected_attacks.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Total Detections</div>
            <div className="text-xs text-green-600 mt-1">Real-time data</div>
          </div>
          <div className="bg-gradient-to-br from-purple-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">{stats.total_packets > 0 ? ((stats.detected_attacks / stats.total_packets) * 100).toFixed(2) : '0'}%</div>
            <div className="text-sm text-gray-600">Detection Rate</div>
            <div className="text-xs text-green-600 mt-1">Based on packets</div>
          </div>
          <div className="bg-gradient-to-br from-green-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">{stats.arp_packets.toLocaleString()}</div>
            <div className="text-sm text-gray-600">ARP Packets</div>
            <div className="text-xs text-green-600 mt-1">Analyzed</div>
          </div>
          <div className="bg-gradient-to-br from-pink-100 to-purple-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-red-600 mb-2">{stats.monitoring_status}</div>
            <div className="text-sm text-gray-600">Status</div>
            <div className="text-xs text-blue-600 mt-1">Current</div>
          </div>
        </div>

        {/* Detailed Stats */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Detection Trends */}
          <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
            <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
            <h2 className="text-2xl font-bold text-blue-700 mb-6">Detection Overview</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                <div>
                  <div className="font-semibold text-blue-800">Total Packets</div>
                  <div className="text-sm text-blue-600">Analyzed packets</div>
                </div>
                <div className="text-2xl font-bold text-blue-600">{stats.total_packets.toLocaleString()}</div>
              </div>
              <div className="flex justify-between items-center p-4 bg-purple-50 rounded-lg">
                <div>
                  <div className="font-semibold text-purple-800">ARP Packets</div>
                  <div className="text-sm text-purple-600">ARP-specific packets</div>
                </div>
                <div className="text-2xl font-bold text-purple-600">{stats.arp_packets.toLocaleString()}</div>
              </div>
              <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                <div>
                  <div className="font-semibold text-green-800">Detected Attacks</div>
                  <div className="text-sm text-green-600">Threats identified</div>
                </div>
                <div className="text-2xl font-bold text-green-600">{stats.detected_attacks.toLocaleString()}</div>
              </div>
            </div>
          </div>
          
          {/* Performance Metrics */}
          <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
            <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
            <h2 className="text-2xl font-bold text-purple-700 mb-6">Performance Metrics</h2>
            <div className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">Detection Rate</span>
                  <span className="text-sm font-medium text-gray-700">
                    {stats.total_packets > 0 ? ((stats.detected_attacks / stats.total_packets) * 100).toFixed(2) : '0'}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full" 
                    style={{ width: `${Math.min((stats.detected_attacks / Math.max(stats.total_packets, 1)) * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">ARP Packet Ratio</span>
                  <span className="text-sm font-medium text-gray-700">
                    {stats.total_packets > 0 ? ((stats.arp_packets / stats.total_packets) * 100).toFixed(1) : '0'}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full" 
                    style={{ width: `${Math.min((stats.arp_packets / Math.max(stats.total_packets, 1)) * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">Monitoring Status</span>
                  <span className="text-sm font-medium text-gray-700">{stats.monitoring_status}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${stats.monitoring_status === 'running' ? 'bg-gradient-to-r from-green-500 to-blue-500' : 'bg-gradient-to-r from-red-500 to-orange-500'}`}
                    style={{ width: stats.monitoring_status === 'running' ? '100%' : '0%' }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Detections Table */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Recent Detections</h2>
          <div className="overflow-x-auto">
            {recentDetections.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">📊</div>
                <p className="text-gray-500">No recent detections found.</p>
              </div>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Time</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Source IP</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Target IP</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Threat Level</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">ML Confidence</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Rule Detection</th>
                  </tr>
                </thead>
                <tbody>
                  {recentDetections.map((detection) => (
                    <tr key={detection.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-sm text-gray-600">
                        {new Date(detection.timestamp).toLocaleString()}
                      </td>
                      <td className="py-3 px-4 text-sm font-mono text-gray-800">
                        {detection.src_ip}
                      </td>
                      <td className="py-3 px-4 text-sm font-mono text-gray-800">
                        {detection.dst_ip}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getSeverityColor(detection.threat_level)}`}>
                          {detection.threat_level}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">
                        {(detection.ml_confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${detection.rule_detection ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                          {detection.rule_detection ? 'Yes' : 'No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics; 