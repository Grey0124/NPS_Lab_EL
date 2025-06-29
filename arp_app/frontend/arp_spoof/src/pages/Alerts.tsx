import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import notificationService from '../services/notificationService';
import websocketService from '../services/websocket';
import type { Alert, AlertStats } from '../services/api';

const Alerts: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [stats, setStats] = useState<AlertStats>({
    total: 0,
    new: 0,
    acknowledged: 0,
    resolved: 0,
    dismissed: 0,
    critical: 0,
    high: 0,
    medium: 0,
    low: 0
  });
  const [filter, setFilter] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Notification settings
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [notificationSettings, setNotificationSettings] = useState({
    critical: true,
    high: true,
    medium: true,
    low: false,
    requireInteraction: true,
    sound: true
  });
  const [showNotificationSettings, setShowNotificationSettings] = useState(false);
  
  // WebSocket status
  const [websocketStatus, setWebsocketStatus] = useState('disconnected');
  const [realtimeEnabled, setRealtimeEnabled] = useState(true);

  useEffect(() => {
    loadAlerts();
    checkNotificationPermission();
    setupWebSocket();
    
    // Cleanup on unmount
    return () => {
      websocketService.disconnect();
      window.removeEventListener('newAlert', handleNewAlert);
      window.removeEventListener('statsUpdate', handleStatsUpdate);
    };
  }, []);

  // Add custom scrollbar styles
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .custom-scrollbar::-webkit-scrollbar {
        width: 8px;
      }
      .custom-scrollbar::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
      }
      .custom-scrollbar::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
        transition: background 0.2s ease;
      }
      .custom-scrollbar::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
      }
      .custom-scrollbar {
        scrollbar-width: thin;
        scrollbar-color: #cbd5e1 #f1f5f9;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const setupWebSocket = async () => {
    try {
      await websocketService.connect();
      setWebsocketStatus(websocketService.getStatus());
      
      // Listen for real-time events
      window.addEventListener('newAlert', handleNewAlert);
      window.addEventListener('statsUpdate', handleStatsUpdate);
      
      // Update status periodically
      const statusInterval = setInterval(() => {
        setWebsocketStatus(websocketService.getStatus());
      }, 5000);
      
      return () => clearInterval(statusInterval);
    } catch (error) {
      console.error('Failed to setup WebSocket:', error);
      setWebsocketStatus('disconnected');
    }
  };

  const handleNewAlert = (event: Event) => {
    const customEvent = event as CustomEvent;
    const newAlert = customEvent.detail;
    console.log('New alert received via WebSocket:', newAlert);
    
    // Add new alert to the list
    setAlerts(prev => [newAlert, ...prev]);
    
    // Update stats
    setStats(prev => ({
      ...prev,
      total: prev.total + 1,
      new: prev.new + 1,
      [newAlert.severity]: prev[newAlert.severity as keyof AlertStats] + 1
    }));
    
    // Show desktop notification if enabled and severity matches settings
    if (notificationsEnabled && notificationSettings[newAlert.severity as keyof typeof notificationSettings]) {
      notificationService.showSecurityAlert(newAlert);
    }
  };

  const handleStatsUpdate = (event: Event) => {
    const customEvent = event as CustomEvent;
    const updatedStats = customEvent.detail;
    console.log('Stats update received via WebSocket:', updatedStats);
    setStats(updatedStats);
  };

  const checkNotificationPermission = async () => {
    const hasPermission = await notificationService.requestPermission();
    setNotificationsEnabled(hasPermission);
  };

  const loadAlerts = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Load alerts and stats from backend
      const [alertsResponse, statsResponse] = await Promise.all([
        apiService.getAlerts(50),
        apiService.getAlertStats()
      ]);
      
      setAlerts(alertsResponse.alerts);
      setStats(statsResponse);
      
      // Show notifications for new alerts if enabled
      if (notificationsEnabled && alertsResponse.alerts.length > 0) {
        const newAlerts = alertsResponse.alerts.filter(alert => 
          alert.status === 'new' && 
          notificationSettings[alert.severity as keyof typeof notificationSettings]
        );
        
        newAlerts.forEach(alert => {
          notificationService.showSecurityAlert(alert);
        });
      }
      
    } catch (error) {
      console.error('Failed to load alerts:', error);
      setError('Failed to load alerts from the backend');
    } finally {
      setIsLoading(false);
    }
  };

  const testNotification = async () => {
    console.log('Test Notification button clicked');
    
    // Check browser support first
    if (!('Notification' in window)) {
      alert('Desktop notifications are not supported in this browser.');
      return;
    }

    // Check current permission status
    console.log('Current notification permission:', Notification.permission);
    
    if (!notificationsEnabled) {
      const granted = await notificationService.requestPermission();
      setNotificationsEnabled(granted);
      if (!granted) {
        alert('Please enable notifications in your browser settings. You can do this by:\n1. Click the lock/info icon in the address bar\n2. Set "Notifications" to "Allow"\n3. Refresh the page and try again.');
        return;
      }
    }

    // Check if page is focused (some browsers don't show notifications when focused)
    const isPageFocused = document.hasFocus();
    const isPageVisible = document.visibilityState === 'visible';
    
    console.log('Page focused:', isPageFocused);
    console.log('Page visible:', isPageVisible);

    // Show a helpful message if page is focused
    if (isPageFocused) {
      console.log('Page is focused - some browsers may not show notifications when the page is active');
    }

    const result = await notificationService.showTestNotification();
    if (!result) {
      alert('Failed to show test notification. Please check:\n1. Browser notification permissions\n2. Windows notification settings\n3. Focus assist settings\n4. Try switching to another tab and testing again');
    } else {
      console.log('Test notification shown successfully');
      
      // Show a success message with additional info
      const message = `Test notification sent successfully!\n\nIf you don't see it:\n1. Check your system notification area\n2. Try switching to another tab\n3. Check Windows Focus Assist settings\n4. Ensure notifications are enabled in Windows settings`;
      alert(message);
    }
  };

  const toggleNotificationSettings = () => {
    setShowNotificationSettings(!showNotificationSettings);
  };

  const updateNotificationSetting = (setting: string, value: boolean) => {
    setNotificationSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const toggleRealtime = () => {
    setRealtimeEnabled(!realtimeEnabled);
    if (!realtimeEnabled) {
      websocketService.connect();
    } else {
      websocketService.disconnect();
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      // TODO: Implement acknowledge alert API call
      // await apiService.acknowledgeAlert(alertId);
      
      // For now, update locally
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'acknowledged', acknowledgedBy: 'current_user' }
          : alert
      ));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const resolveAlert = async (alertId: string) => {
    try {
      // TODO: Implement resolve alert API call
      // await apiService.resolveAlert(alertId);
      
      // For now, update locally
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'resolved', resolvedAt: new Date().toISOString() }
          : alert
      ));
    } catch (error) {
      console.error('Failed to resolve alert:', error);
    }
  };

  const dismissAlert = async (alertId: string) => {
    try {
      // TODO: Implement dismiss alert API call
      // await apiService.dismissAlert(alertId);
      
      // For now, update locally
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'dismissed' }
          : alert
      ));
    } catch (error) {
      console.error('Failed to dismiss alert:', error);
    }
  };

  const clearAllAlerts = async () => {
    if (!confirm('Are you sure you want to clear all alerts? This action cannot be undone.')) {
      return;
    }

    try {
      const result = await apiService.clearAlerts();
      
      if (result.status === 'success') {
        setAlerts([]);
        setStats({
          total: 0,
          new: 0,
          acknowledged: 0,
          resolved: 0,
          dismissed: 0,
          critical: 0,
          high: 0,
          medium: 0,
          low: 0
        });
      } else {
        setError(result.message || 'Failed to clear alerts');
      }
    } catch (error) {
      console.error('Failed to clear alerts:', error);
      setError('Failed to clear alerts');
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'arp_spoofing': return '🚨';
      case 'suspicious_activity': return '⚠️';
      case 'system_error': return '🔧';
      case 'configuration_change': return '⚙️';
      default: return '📢';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'bg-blue-100 text-blue-800';
      case 'acknowledged': return 'bg-yellow-100 text-yellow-800';
      case 'resolved': return 'bg-green-100 text-green-800';
      case 'dismissed': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getWebSocketStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-green-500';
      case 'connecting': return 'bg-yellow-500';
      case 'disconnected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true;
    return alert.status === filter;
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading alerts...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="bg-white/80 border border-red-200 rounded-xl shadow-xl p-8 text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-red-700 mb-4">Error Loading Alerts</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={loadAlerts}
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
          <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-2">Security Alerts</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">Monitor and manage security alerts from your ARP spoofing detection system. Stay informed about potential threats and system events.</p>
        </div>

        {/* Notification Settings Panel */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-purple-700 mb-2">🔔 Desktop Notifications</h2>
              <p className="text-gray-600">Configure desktop notifications for security alerts</p>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={testNotification}
                className="bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow hover:from-green-600 hover:to-green-700 transition-all"
              >
                Test Notification
              </button>
              <button
                onClick={toggleNotificationSettings}
                className="bg-gradient-to-r from-purple-500 to-purple-600 text-white font-semibold py-2 px-4 rounded-lg shadow hover:from-purple-600 hover:to-purple-700 transition-all"
              >
                {showNotificationSettings ? 'Hide Settings' : 'Show Settings'}
              </button>
            </div>
          </div>

          {/* Connection Status */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {/* Notification Status */}
            <div className="flex items-center space-x-4">
              <div className={`w-4 h-4 rounded-full ${notificationsEnabled ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {notificationsEnabled ? 'Notifications Enabled' : 'Notifications Disabled'}
              </span>
              {!notificationsEnabled && (
                <button
                  onClick={checkNotificationPermission}
                  className="text-blue-600 hover:text-blue-800 text-sm font-medium underline"
                >
                  Enable Notifications
                </button>
              )}
            </div>

            {/* WebSocket Status */}
            <div className="flex items-center space-x-4">
              <div className={`w-4 h-4 rounded-full ${getWebSocketStatusColor(websocketStatus)}`}></div>
              <span className="text-sm font-medium">
                Real-time: {websocketStatus.charAt(0).toUpperCase() + websocketStatus.slice(1)}
              </span>
              <button
                onClick={toggleRealtime}
                className={`text-sm font-medium px-3 py-1 rounded ${
                  realtimeEnabled 
                    ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                    : 'bg-green-100 text-green-600 hover:bg-green-200'
                }`}
              >
                {realtimeEnabled ? 'Disable' : 'Enable'}
              </button>
            </div>
          </div>

          {/* Notification Settings */}
          {showNotificationSettings && (
            <div className="bg-gray-50 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Notification Preferences</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <h4 className="font-medium text-gray-700">Alert Severity</h4>
                  {Object.entries(notificationSettings).filter(([key]) => ['critical', 'high', 'medium', 'low'].includes(key)).map(([severity, enabled]) => (
                    <label key={severity} className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={(e) => updateNotificationSetting(severity, e.target.checked)}
                        className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                      />
                      <span className="text-sm font-medium capitalize">{severity}</span>
                    </label>
                  ))}
                </div>
                
                <div className="space-y-3">
                  <h4 className="font-medium text-gray-700">Notification Behavior</h4>
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={notificationSettings.requireInteraction}
                      onChange={(e) => updateNotificationSetting('requireInteraction', e.target.checked)}
                      className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="text-sm">Require user interaction</span>
                  </label>
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={notificationSettings.sound}
                      onChange={(e) => updateNotificationSetting('sound', e.target.checked)}
                      className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="text-sm">Play notification sound</span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Alert Statistics */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">{stats.total}</div>
            <div className="text-sm text-gray-600">Total Alerts</div>
          </div>
          <div className="bg-gradient-to-br from-purple-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-red-600 mb-2">{(stats.critical || 0) + (stats.high || 0)}</div>
            <div className="text-sm text-gray-600">High Priority</div>
          </div>
          <div className="bg-gradient-to-br from-pink-100 to-purple-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-yellow-600 mb-2">{stats.new}</div>
            <div className="text-sm text-gray-600">New Alerts</div>
          </div>
          <div className="bg-gradient-to-br from-green-100 to-blue-100 rounded-xl p-6 shadow text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">{stats.resolved}</div>
            <div className="text-sm text-gray-600">Resolved</div>
          </div>
        </div>

        {/* Filter and Actions */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4 absolute left-0 right-0 top-0"></div>
          <div className="flex items-center space-x-4">
            <label className="text-sm font-semibold text-gray-700">Filter:</label>
            <select value={filter} onChange={(e) => setFilter(e.target.value)} className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent">
              <option value="all">All Alerts ({stats.total})</option>
              <option value="new">New ({stats.new})</option>
              <option value="acknowledged">Acknowledged ({stats.acknowledged})</option>
              <option value="resolved">Resolved ({stats.resolved})</option>
              <option value="dismissed">Dismissed ({stats.dismissed})</option>
            </select>
          </div>
          <div className="flex gap-2">
            <button onClick={loadAlerts} className="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-2 px-6 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 scale-105 transition-all">Refresh</button>
            <button onClick={clearAllAlerts} className="border-2 border-red-500 text-red-600 rounded-lg font-semibold py-2 px-6 hover:bg-red-50 transition-all">Clear All</button>
          </div>
        </div>

        {/* Alerts List */}
        <div className="mb-8">
          <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-6">
            <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
            <h2 className="text-2xl font-bold text-purple-700 mb-4">Security Alerts</h2>
            
            {/* Scrollable alerts container */}
            <div className="max-h-96 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
              {filteredAlerts.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🎉</div>
                  <h3 className="text-xl font-semibold text-blue-700 mb-2">No Alerts</h3>
                  <p className="text-gray-600">{filter === 'all' ? 'Great! No security alerts at the moment.' : `No ${filter} alerts found.`}</p>
                </div>
              ) : (
                filteredAlerts.map((alert) => (
                  <div key={alert.id} className="bg-white border border-purple-200 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4 flex-1">
                        <div className="text-2xl">{getTypeIcon(alert.type)}</div>
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center space-x-3">
                            <h3 className="text-lg font-semibold text-blue-700">{alert.title}</h3>
                            <span className={`px-2 py-1 rounded-full text-xs font-semibold border ${getSeverityColor(alert.severity)}`}>{alert.severity}</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(alert.status)}`}>{alert.status}</span>
                          </div>
                          <p className="text-gray-600">{alert.description}</p>
                          {(alert.sourceIP || alert.targetIP) && (
                            <div className="text-sm text-gray-500 space-x-4">
                              {alert.sourceIP && <span>Source: <code className="bg-gray-100 px-1 rounded">{alert.sourceIP}</code></span>}
                              {alert.targetIP && <span>Target: <code className="bg-gray-100 px-1 rounded">{alert.targetIP}</code></span>}
                            </div>
                          )}
                          <div className="text-sm text-gray-500">
                            {new Date(alert.timestamp).toLocaleString()}
                            {alert.acknowledgedBy && ` • Acknowledged by ${alert.acknowledgedBy}`}
                            {alert.resolvedAt && ` • Resolved at ${new Date(alert.resolvedAt).toLocaleString()}`}
                          </div>
                        </div>
                      </div>
                      <div className="flex flex-col space-y-2 ml-4">
                        {alert.status === 'new' && (
                          <>
                            <button onClick={() => acknowledgeAlert(alert.id)} className="bg-gradient-to-r from-yellow-400 to-yellow-600 text-white rounded text-sm px-3 py-1 font-semibold hover:from-yellow-500 hover:to-yellow-700 transition-colors">Acknowledge</button>
                            <button onClick={() => resolveAlert(alert.id)} className="bg-gradient-to-r from-green-500 to-green-700 text-white rounded text-sm px-3 py-1 font-semibold hover:from-green-600 hover:to-green-800 transition-colors">Resolve</button>
                          </>
                        )}
                        {alert.status === 'acknowledged' && (
                          <button onClick={() => resolveAlert(alert.id)} className="bg-gradient-to-r from-green-500 to-green-700 text-white rounded text-sm px-3 py-1 font-semibold hover:from-green-600 hover:to-green-800 transition-colors">Resolve</button>
                        )}
                        {alert.status !== 'resolved' && alert.status !== 'dismissed' && (
                          <button onClick={() => dismissAlert(alert.id)} className="border border-gray-300 text-gray-600 rounded text-sm px-3 py-1 font-semibold hover:bg-gray-50 transition-colors">Dismiss</button>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            
            {/* Alerts count and pagination info */}
            <div className="mt-4 pt-4 border-t border-gray-200 flex justify-between items-center text-sm text-gray-600">
              <span>Showing {filteredAlerts.length} of {stats.total} alerts</span>
              {filteredAlerts.length > 0 && (
                <span>Scroll to see more alerts</span>
              )}
            </div>
          </div>
        </div>

        {/* Alert History Summary */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Alert History Summary</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600 mb-1">{stats.critical}</div>
              <div className="text-sm text-gray-600">Critical</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600 mb-1">{stats.high}</div>
              <div className="text-sm text-gray-600">High</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600 mb-1">{stats.medium}</div>
              <div className="text-sm text-gray-600">Medium</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600 mb-1">{stats.low}</div>
              <div className="text-sm text-gray-600">Low</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Alerts; 