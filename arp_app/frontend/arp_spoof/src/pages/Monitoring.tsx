import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import type { 
  MonitoringStatus, 
  DetectionRecord, 
  WebSocketMessage,
  NetworkInterface 
} from '../services/api';

const Monitoring: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [selectedInterface, setSelectedInterface] = useState('');
  const [interfaces, setInterfaces] = useState<NetworkInterface[]>([]);
  const [status, setStatus] = useState<MonitoringStatus>({
    is_monitoring: false,
    current_interface: null,
    live_stats: {
      total_packets: 0,
      arp_packets: 0,
      detected_attacks: 0,
      last_attack_time: null,
      current_interface: null,
      monitoring_status: 'stopped',
      // Prevention statistics
      prevention_active: false,
      packets_dropped: 0,
      arp_entries_corrected: 0,
      quarantined_ips: 0,
      rate_limited_ips: 0
    },
    recent_detections_count: 0
  });
  const [recentDetections, setRecentDetections] = useState<DetectionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAction, setCurrentAction] = useState<'none' | 'starting' | 'stopping'>('none');
  const [backendAvailable, setBackendAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize component
  useEffect(() => {
    initializeMonitoring();
    
    // Set up periodic status refresh
    const statusInterval = setInterval(() => {
      if (backendAvailable && isMonitoring) {
        loadMonitoringStatus();
        loadRecentDetections(); // Also refresh recent detections
      }
    }, 5000); // Refresh every 5 seconds when monitoring
    
    return () => {
      // Cleanup WebSocket connection
      apiService.disconnectWebSocket();
      // Clear interval
      clearInterval(statusInterval);
    };
  }, [backendAvailable, isMonitoring]);

  const initializeMonitoring = async () => {
    try {
      // First test basic connectivity with ping
      try {
        const pingResponse = await apiService.ping();
        console.log('Backend ping successful:', pingResponse);
      } catch (pingError) {
        console.error('Backend ping failed:', pingError);
        setError('Backend server is not responding. Please ensure the FastAPI server is running on port 8000.');
        setBackendAvailable(false);
        return;
      }

      // Check if backend is available
      const available = await apiService.isBackendAvailable();
      setBackendAvailable(available);
      
      if (!available) {
        setError('Backend service is not available. Please ensure the FastAPI server is running.');
        return;
      }

      // Load network interfaces
      await loadInterfaces();
      
      // Load current monitoring status
      await loadMonitoringStatus();
      
      // Load recent detections
      await loadRecentDetections();
      
      // Connect to WebSocket for real-time updates
      apiService.connectWebSocket(handleWebSocketMessage);
      
    } catch (error) {
      console.error('Failed to initialize monitoring:', error);
      setError('Failed to connect to monitoring service');
    }
  };

  const retryConnection = async () => {
    console.log('Retrying connection...');
    setError(null);
    setIsLoading(true);
    
    try {
      // Disconnect existing WebSocket
      apiService.disconnectWebSocket();
      
      // Wait a moment before reconnecting
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Reinitialize everything
      await initializeMonitoring();
      
    } catch (error) {
      console.error('Failed to retry connection:', error);
      setError('Failed to reconnect to monitoring service');
    } finally {
      setIsLoading(false);
    }
  };

  const loadInterfaces = async () => {
    try {
      const response = await apiService.getNetworkInterfaces();
      const interfaceList = response.interfaces.map(name => ({
        name,
        description: `Network Interface ${name}`,
        status: 'up' as const
      }));
      setInterfaces(interfaceList);
    } catch (error) {
      console.error('Failed to load interfaces:', error);
      setError('Failed to load network interfaces');
    }
  };

  const loadMonitoringStatus = async () => {
    try {
      const statusData = await apiService.getMonitoringStatus();
      setStatus(statusData);
      setIsMonitoring(statusData.is_monitoring);
    } catch (error) {
      console.error('Failed to load monitoring status:', error);
    }
  };

  const loadRecentDetections = async () => {
    try {
      // Test detections endpoint - COMMENTED OUT: Automatic registry addition is now working properly
      // try {
      //   const testResponse = await apiService.testDetectionsEndpoint();
      //   console.log('Test detections response:', testResponse);
      //   
      //   if (testResponse.status !== 'success') {
      //     console.warn('Test detections endpoint failed:', testResponse.message);
      //     setRecentDetections([]);
      //     return;
      //   }
      // } catch (testError) {
      //   console.warn('Test detections endpoint error:', testError);
      //   setRecentDetections([]);
      //   return;
      // }
      
      // Load actual detections directly
      const response = await apiService.getRecentDetections(10);
      setRecentDetections(response.detections);
    } catch (error) {
      console.error('Failed to load recent detections:', error);
      // Don't set error state for this, just log it and continue with empty detections
      setRecentDetections([]);
    }
  };

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    console.log('WebSocket message received:', message); // Debug logging
    
    switch (message.type) {
      case 'monitoring_status':
        console.log('Processing monitoring status:', message.status); // Debug logging
        if (message.status === 'started') {
          console.log('Setting isMonitoring to true'); // Debug logging
          setIsMonitoring(true);
          setStatus(prev => ({
            ...prev,
            is_monitoring: true,
            current_interface: message.interface || null
          }));
        } else if (message.status === 'stopped') {
          console.log('Setting isMonitoring to false'); // Debug logging
          setIsMonitoring(false);
          setStatus(prev => ({
            ...prev,
            is_monitoring: false,
            current_interface: null
          }));
        }
        break;
      
      case 'attack_detected':
        console.log('Attack detected:', message.data); // Debug logging
        setRecentDetections(prev => [message.data, ...prev.slice(0, 9)]);
        setStatus(prev => {
          const newStatus = {
            ...prev,
            live_stats: {
              ...prev.live_stats,
              detected_attacks: prev.live_stats.detected_attacks + 1,
              last_attack_time: message.data.timestamp
            }
          };
          console.log('Updated threat count to:', newStatus.live_stats.detected_attacks); // Debug logging
          return newStatus;
        });
        break;
      
      case 'stats_update':
        console.log('Stats update:', message.data); // Debug logging
        setStatus(prev => {
          const newStatus = {
            ...prev,
            live_stats: {
              ...prev.live_stats,
              ...message.data
            }
          };
          console.log('Updated stats, new threat count:', newStatus.live_stats.detected_attacks); // Debug logging
          return newStatus;
        });
        break;
        
      default:
        console.log('Unknown message type:', message.type); // Debug logging
    }
  };

  const startMonitoring = async () => {
    if (!selectedInterface) {
      setError('Please select a network interface');
      return;
    }

    setIsLoading(true);
    setCurrentAction('starting');
    setError(null);
    
    try {
      const result = await apiService.startMonitoring(selectedInterface);
      
      if (result.status === 'success') {
        setIsMonitoring(true);
        await loadMonitoringStatus();
      } else {
        setError(result.message || 'Failed to start monitoring');
      }
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      setError('Failed to start monitoring. Please check your network configuration.');
    } finally {
      setIsLoading(false);
      setCurrentAction('none');
    }
  };

  const stopMonitoring = async () => {
    console.log('Stop monitoring called, isLoading:', isLoading); // Debug logging
    setIsLoading(true);
    setCurrentAction('stopping');
    setError(null);
    
    try {
      console.log('Calling apiService.stopMonitoring()...'); // Debug logging
      const result = await apiService.stopMonitoring();
      console.log('Stop monitoring result:', result); // Debug logging
      
      if (result.status === 'success') {
        console.log('Setting isMonitoring to false'); // Debug logging
        setIsMonitoring(false);
        await loadMonitoringStatus();
      } else {
        setError(result.message || 'Failed to stop monitoring');
      }
    } catch (error) {
      console.error('Failed to stop monitoring:', error);
      setError('Failed to stop monitoring');
    } finally {
      console.log('Setting isLoading to false'); // Debug logging
      setIsLoading(false);
      setCurrentAction('none');
    }
  };

  if (!backendAvailable) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="bg-white/80 border border-red-200 rounded-xl shadow-xl p-8 text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-red-700 mb-4">Backend Unavailable</h2>
          <p className="text-gray-600 mb-4">{error || 'The monitoring service is not available.'}</p>
          <button 
            onClick={retryConnection}
            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-2 px-6 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 transition-all"
          >
            Retry Connection
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
          <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-2">Network Monitoring</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">Monitor your network for ARP spoofing attacks in real-time. Select an interface and start monitoring to detect potential threats.</p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mb-8">
            <div className="flex items-center">
              <span className="text-xl mr-2">⚠️</span>
              <span>{error}</span>
              <button 
                onClick={() => setError(null)}
                className="ml-auto text-red-500 hover:text-red-700"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Status Card */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-blue-700">Monitoring Status</h2>
            <div className={`px-4 py-2 rounded-full text-sm font-semibold ${isMonitoring ? 'bg-gradient-to-r from-blue-400 to-green-400 text-white' : 'bg-gradient-to-r from-purple-400 to-pink-400 text-white'}`}>
              {isMonitoring ? 'Active' : 'Inactive'}
            </div>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">{status.live_stats.total_packets.toLocaleString()}</div>
              <div className="text-sm text-gray-600">Packets Analyzed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-600 mb-2">{status.live_stats.detected_attacks}</div>
              <div className="text-sm text-gray-600">Threats Detected</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">{status.live_stats.packets_dropped}</div>
              <div className="text-sm text-gray-600">Packets Dropped</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">{status.live_stats.arp_entries_corrected}</div>
              <div className="text-sm text-gray-600">ARP Entries Corrected</div>
              </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">{status.live_stats.quarantined_ips}</div>
              <div className="text-sm text-gray-600">Quarantined IPs</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-indigo-600 mb-2">{status.live_stats.rate_limited_ips}</div>
              <div className="text-sm text-gray-600">Rate Limited IPs</div>
            </div>
          </div>
          
          {/* Prevention Status */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-700">Prevention Status</h3>
              <div className={`px-3 py-1 rounded-full text-sm font-semibold ${status.live_stats.prevention_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                {status.live_stats.prevention_active ? 'Active' : 'Inactive'}
              </div>
            </div>
            <div className="mt-4 grid md:grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Current Interface:</span>
                <span className="text-sm font-medium text-gray-900 truncate max-w-xs" title={status.current_interface || 'None'}>
                  {status.current_interface || 'None'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Recent Activity:</span>
                <span className="text-sm font-medium text-gray-900">
                  {status.live_stats.last_attack_time ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Control Panel</h2>
          <div className="space-y-6">
            {/* Interface Selection */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Network Interface</label>
              <select 
                value={selectedInterface} 
                onChange={(e) => setSelectedInterface(e.target.value)} 
                disabled={isMonitoring} 
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
              >
                <option value="">Select an interface</option>
                {interfaces.map((iface) => (
                  <option key={iface.name} value={iface.name}>
                    {iface.name} - {iface.description}
                  </option>
                ))}
              </select>
            </div>
            
            {/* Control Buttons */}
            <div className="flex gap-4">
              {!isMonitoring ? (
                <button 
                  onClick={startMonitoring} 
                  disabled={!selectedInterface || isLoading} 
                  className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-3 px-8 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {currentAction === 'starting' ? 'Starting...' : 'Start Monitoring'}
                </button>
              ) : (
                <button 
                  onClick={stopMonitoring} 
                  disabled={isLoading} 
                  className="flex-1 bg-gradient-to-r from-red-500 to-pink-500 text-white font-semibold py-3 px-8 rounded-lg shadow hover:from-red-600 hover:to-pink-600 scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {currentAction === 'stopping' ? 'Stopping...' : 'Stop Monitoring'}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Live Detection Feed */}
        <div className="bg-black border border-gray-800 rounded-xl shadow-xl p-0 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-0"></div>
          <div className="flex items-center justify-between px-8 py-4">
            <h2 className="text-xl font-mono font-bold text-green-400">Live Detection Feed</h2>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
              <span className="text-sm text-gray-400 font-mono">{isMonitoring ? 'Live' : 'Offline'}</span>
            </div>
          </div>
          <div
            className="overflow-y-auto px-8 py-4 font-mono text-sm"
            style={{ maxHeight: '320px', background: '#18181b' }}
          >
            {recentDetections.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <div className="text-5xl mb-4">🔍</div>
                <p>
                  {isMonitoring ? 'Monitoring for threats... No detections yet.' : 'Start monitoring to see detection results here.'}
                </p>
              </div>
            ) : (
              recentDetections.map((detection, index) => (
                <div
                  key={index}
                  className={`flex items-center space-x-4 p-2 border-b border-gray-700 last:border-b-0 ${
                    detection.threat_level === 'HIGH'
                      ? 'bg-red-900/40 text-red-300'
                      : detection.threat_level === 'MEDIUM'
                      ? 'bg-yellow-900/30 text-yellow-200'
                      : 'text-gray-200'
                  }`}
                >
                  <span className="font-bold text-green-400">{new Date(detection.timestamp).toLocaleTimeString()}</span>
                  <span>
                    <span className="font-bold text-cyan-300">{detection.src_ip}</span>
                    {' → '}
                    <span className="font-bold text-blue-300">{detection.dst_ip}</span>
                  </span>
                  <span>
                    <span className="font-bold">{detection.threat_level}</span>
                    {' | '}
                    ML: {detection.ml_confidence == null ? 'N/A' : (detection.ml_confidence * 100).toFixed(1) + '%'}
                    {' | '}
                    Rule: {detection.rule_detection ? 'Yes' : 'No'}
                  </span>
                  <span className="truncate text-xs text-gray-400">{detection.rule_reason}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl p-6 shadow text-center">
            <div className="text-4xl font-bold text-blue-600 mb-2">24/7</div>
            <div className="text-sm text-gray-600">Continuous Monitoring</div>
          </div>
          <div className="bg-gradient-to-br from-purple-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-4xl font-bold text-purple-600 mb-2">&lt;1s</div>
            <div className="text-sm text-gray-600">Response Time</div>
          </div>
          <div className="bg-gradient-to-br from-green-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-4xl font-bold text-green-600 mb-2">99.9%</div>
            <div className="text-sm text-gray-600">Detection Accuracy</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Monitoring; 