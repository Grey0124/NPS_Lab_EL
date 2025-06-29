import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import type { DetectionConfig, AlertConfig, SystemConfig } from '../services/api';

const Configuration: React.FC = () => {
  const [detectionConfig, setDetectionConfig] = useState<DetectionConfig>({
    sensitivity: 'medium',
    scanInterval: 5,
    maxRetries: 3,
    timeout: 30,
    enableML: true,
    enableHeuristics: true
  });

  const [alertConfig, setAlertConfig] = useState<AlertConfig>({
    emailEnabled: false,
    emailRecipients: [],
    webhookEnabled: false,
    webhookUrl: '',
    notificationCooldown: 300,
    enableDesktopNotifications: true,
    enableSoundAlerts: true
  });

  const [systemConfig, setSystemConfig] = useState<SystemConfig>({
    autoStart: false,
    logLevel: 'info',
    maxLogSize: 100,
    enableBackup: true,
    backupInterval: 24
  });

  const [isLoading, setIsLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const config = await apiService.getConfiguration();
      
      // The API service now handles the conversion from backend to frontend format
      setDetectionConfig(config.detection);
      setAlertConfig(config.alerts);
      setSystemConfig(config.system);
      
    } catch (error) {
      console.error('Failed to load configuration:', error);
      setError('Failed to load configuration from the backend');
    } finally {
      setIsLoading(false);
    }
  };

  const saveConfiguration = async () => {
    setSaveStatus('saving');
    setError(null);
    
    try {
      // The API service now handles the conversion from frontend to backend format
      const result = await apiService.updateConfiguration({
        detection: detectionConfig,
        alerts: alertConfig
      });
      
      if (result.status === 'success') {
        setSaveStatus('success');
        setTimeout(() => setSaveStatus('idle'), 3000);
      } else {
        setSaveStatus('error');
        setError(result.message || 'Failed to save configuration');
        setTimeout(() => setSaveStatus('idle'), 3000);
      }
      
    } catch (error) {
      console.error('Failed to save configuration:', error);
      setSaveStatus('error');
      setError('Failed to save configuration. Please check your connection.');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };

  const resetConfiguration = async () => {
    if (!confirm('Are you sure you want to reset all configuration to defaults?')) {
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiService.resetConfiguration();
      
      if (result.status === 'success') {
        // Reload configuration after reset
        await loadConfiguration();
      } else {
        setError(result.message || 'Failed to reset configuration');
      }
      
    } catch (error) {
      console.error('Failed to reset configuration:', error);
      setError('Failed to reset configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const addEmailRecipient = () => {
    const email = prompt('Enter email address:');
    if (email && email.includes('@')) {
      setAlertConfig(prev => ({
        ...prev,
        emailRecipients: [...prev.emailRecipients, email]
      }));
    }
  };

  const removeEmailRecipient = (index: number) => {
    setAlertConfig(prev => ({
      ...prev,
      emailRecipients: prev.emailRecipients.filter((_, i) => i !== index)
    }));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading configuration...</p>
        </div>
      </div>
    );
  }

  if (error && !isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-blue-200 flex items-center justify-center">
        <div className="bg-white/80 border border-red-200 rounded-xl shadow-xl p-8 text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-red-700 mb-4">Configuration Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button 
            onClick={loadConfiguration}
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
          <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent drop-shadow mb-2">Configuration</h1>
          <p className="text-gray-600 max-w-2xl mx-auto">Configure detection parameters, alert settings, and system preferences to customize your ARP spoofing detection system.</p>
        </div>

        {/* Save Status */}
        {saveStatus !== 'idle' && (
          <div className={`p-4 rounded-lg mb-8 ${saveStatus === 'success' ? 'bg-green-100 text-green-800' : saveStatus === 'error' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'}`}>
            {saveStatus === 'saving' && 'Saving configuration...'}
            {saveStatus === 'success' && 'Configuration saved successfully!'}
            {saveStatus === 'error' && 'Failed to save configuration. Please try again.'}
          </div>
        )}

        {/* Detection Configuration */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Detection Settings</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Detection Sensitivity
              </label>
              <select
                value={detectionConfig.sensitivity}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, sensitivity: e.target.value as any }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="low">Low - Fewer false positives</option>
                <option value="medium">Medium - Balanced</option>
                <option value="high">High - More sensitive</option>
                <option value="critical">Critical - Maximum sensitivity</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Scan Interval (seconds)
              </label>
              <input
                type="number"
                min="1"
                max="60"
                value={detectionConfig.scanInterval}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, scanInterval: parseInt(e.target.value) }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Max Retries
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={detectionConfig.maxRetries}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, maxRetries: parseInt(e.target.value) }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Timeout (seconds)
              </label>
              <input
                type="number"
                min="5"
                max="120"
                value={detectionConfig.timeout}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, timeout: parseInt(e.target.value) }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableML"
                checked={detectionConfig.enableML}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, enableML: e.target.checked }))}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="enableML" className="ml-2 text-sm text-gray-700">
                Enable Machine Learning Detection
              </label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableHeuristics"
                checked={detectionConfig.enableHeuristics}
                onChange={(e) => setDetectionConfig(prev => ({ ...prev, enableHeuristics: e.target.checked }))}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="enableHeuristics" className="ml-2 text-sm text-gray-700">
                Enable Heuristic Analysis
              </label>
            </div>
          </div>
        </div>

        {/* Alert Configuration */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-purple-700 mb-6">Alert Settings</h2>
          
          <div className="space-y-6">
            {/* Email Alerts */}
            <div>
              <div className="flex items-center mb-4">
                <input
                  type="checkbox"
                  id="emailEnabled"
                  checked={alertConfig.emailEnabled}
                  onChange={(e) => setAlertConfig(prev => ({ ...prev, emailEnabled: e.target.checked }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="emailEnabled" className="ml-2 text-sm font-semibold text-gray-700">
                  Enable Email Alerts
                </label>
              </div>
              
              {alertConfig.emailEnabled && (
                <div className="ml-6 space-y-4">
                  <div>
                    <label className="block text-sm text-gray-600 mb-2">Email Recipients</label>
                    <div className="space-y-2">
                      {alertConfig.emailRecipients.map((email, index) => (
                        <div key={index} className="flex items-center space-x-2">
                          <span className="flex-1 px-3 py-2 bg-gray-100 rounded text-sm">{email}</span>
                          <button
                            onClick={() => removeEmailRecipient(index)}
                            className="px-2 py-1 text-red-600 hover:bg-red-50 rounded"
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                      <button
                        onClick={addEmailRecipient}
                        className="px-4 py-2 text-blue-600 border border-blue-300 rounded hover:bg-blue-50"
                      >
                        + Add Email
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Webhook Alerts */}
            <div>
              <div className="flex items-center mb-4">
                <input
                  type="checkbox"
                  id="webhookEnabled"
                  checked={alertConfig.webhookEnabled}
                  onChange={(e) => setAlertConfig(prev => ({ ...prev, webhookEnabled: e.target.checked }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="webhookEnabled" className="ml-2 text-sm font-semibold text-gray-700">
                  Enable Webhook Alerts
                </label>
              </div>
              
              {alertConfig.webhookEnabled && (
                <div className="ml-6">
                  <label className="block text-sm text-gray-600 mb-2">Webhook URL</label>
                  <input
                    type="url"
                    value={alertConfig.webhookUrl}
                    onChange={(e) => setAlertConfig(prev => ({ ...prev, webhookUrl: e.target.value }))}
                    placeholder="https://your-webhook-url.com/endpoint"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              )}
            </div>

            {/* Notification Settings */}
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Notification Cooldown (seconds)
                </label>
                <input
                  type="number"
                  min="0"
                  max="3600"
                  value={alertConfig.notificationCooldown}
                  onChange={(e) => setAlertConfig(prev => ({ ...prev, notificationCooldown: parseInt(e.target.value) }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="desktopNotifications"
                  checked={alertConfig.enableDesktopNotifications}
                  onChange={(e) => setAlertConfig(prev => ({ ...prev, enableDesktopNotifications: e.target.checked }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="desktopNotifications" className="ml-2 text-sm text-gray-700">
                  Enable Desktop Notifications
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="soundAlerts"
                  checked={alertConfig.enableSoundAlerts}
                  onChange={(e) => setAlertConfig(prev => ({ ...prev, enableSoundAlerts: e.target.checked }))}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="soundAlerts" className="ml-2 text-sm text-gray-700">
                  Enable Sound Alerts
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* System Configuration */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">System Settings</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Log Level
              </label>
              <select
                value={systemConfig.logLevel}
                onChange={(e) => setSystemConfig(prev => ({ ...prev, logLevel: e.target.value as any }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="debug">Debug</option>
                <option value="info">Info</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Max Log Size (MB)
              </label>
              <input
                type="number"
                min="10"
                max="1000"
                value={systemConfig.maxLogSize}
                onChange={(e) => setSystemConfig(prev => ({ ...prev, maxLogSize: parseInt(e.target.value) }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Backup Interval (hours)
              </label>
              <input
                type="number"
                min="1"
                max="168"
                value={systemConfig.backupInterval}
                onChange={(e) => setSystemConfig(prev => ({ ...prev, backupInterval: parseInt(e.target.value) }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="mt-6 space-y-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoStart"
                checked={systemConfig.autoStart}
                onChange={(e) => setSystemConfig(prev => ({ ...prev, autoStart: e.target.checked }))}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="autoStart" className="ml-2 text-sm text-gray-700">
                Auto-start monitoring on system boot
              </label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="enableBackup"
                checked={systemConfig.enableBackup}
                onChange={(e) => setSystemConfig(prev => ({ ...prev, enableBackup: e.target.checked }))}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="enableBackup" className="ml-2 text-sm text-gray-700">
                Enable automatic configuration backup
              </label>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          <button 
            onClick={saveConfiguration} 
            disabled={saveStatus === 'saving'} 
            className="bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold py-3 px-8 rounded-lg shadow hover:from-blue-600 hover:to-purple-700 scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saveStatus === 'saving' ? 'Saving...' : 'Save Configuration'}
          </button>
          <button 
            onClick={resetConfiguration} 
            disabled={isLoading} 
            className="border-2 border-red-500 text-red-600 rounded-lg font-semibold py-3 px-8 hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
};

export default Configuration; 