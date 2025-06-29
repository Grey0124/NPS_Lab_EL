import React, { useState, useEffect } from 'react';
import apiService from '../services/api';
import type { AlertConfig, SystemConfig } from '../services/api';

const Configuration: React.FC = () => {
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
  const [validationResult, setValidationResult] = useState<{ valid: boolean; errors: string[]; warnings: string[] } | null>(null);
  const [backups, setBackups] = useState<Array<{ name: string; path: string; created: string; size: number }>>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [registryEntries, setRegistryEntries] = useState<Record<string, string>>({});
  const [showRegistry, setShowRegistry] = useState(false);
  const [newEntry, setNewEntry] = useState({ ip: '', mac: '' });
  const [registryLastUpdated, setRegistryLastUpdated] = useState<Date | null>(null);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [registryEntryCount, setRegistryEntryCount] = useState(0);
  const [newEntriesDetected, setNewEntriesDetected] = useState(false);

  useEffect(() => {
    loadConfiguration();
    loadBackups();
    loadRegistry();
    
    // Set up automatic registry refresh every 5 seconds
    const registryInterval = setInterval(() => {
      loadRegistry();
    }, 5000);
    
    // Cleanup interval on component unmount
    return () => {
      clearInterval(registryInterval);
    };
  }, []);

  const loadConfiguration = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const config = await apiService.getConfiguration();
      
      // The API service now handles the conversion from backend to frontend format
      setAlertConfig(config.alerts);
      setSystemConfig(config.system);
      
    } catch (error) {
      console.error('Failed to load configuration:', error);
      setError('Failed to load configuration from the backend');
    } finally {
      setIsLoading(false);
    }
  };

  const loadBackups = async () => {
    try {
      const result = await apiService.listConfigurationBackups();
      setBackups(result.backups);
    } catch (error) {
      console.error('Failed to load backups:', error);
    }
  };

  const loadRegistry = async () => {
    setRegistryLoading(true);
    try {
      console.log('🔄 Loading registry entries...');
      const result = await apiService.getRegistryEntries();
      console.log('📊 Registry API response:', result);
      
      const newEntryCount = Object.keys(result.entries).length;
      console.log('📊 New entry count:', newEntryCount);
      console.log('📊 Previous entry count:', registryEntryCount);
      console.log('📊 Registry entries:', result.entries);
      
      // Check if new entries were added
      if (registryEntryCount > 0 && newEntryCount > registryEntryCount) {
        console.log('🆕 New entries detected!');
        setNewEntriesDetected(true);
        // Clear the indicator after 3 seconds
        setTimeout(() => setNewEntriesDetected(false), 3000);
      }
      
      setRegistryEntries(result.entries);
      setRegistryEntryCount(newEntryCount);
      setRegistryLastUpdated(new Date());
      console.log('✅ Registry loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load registry:', error);
    } finally {
      setRegistryLoading(false);
    }
  };

  const validateConfiguration = async () => {
    try {
      const result = await apiService.validateConfiguration();
      setValidationResult({
        valid: result.valid,
        errors: result.errors,
        warnings: result.warnings
      });
    } catch (error) {
      console.error('Failed to validate configuration:', error);
      setError('Failed to validate configuration');
    }
  };

  const saveConfiguration = async () => {
    setSaveStatus('saving');
    setError(null);
    
    try {
      // The API service now handles the conversion from frontend to backend format
      const result = await apiService.updateConfiguration({
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

  const exportConfiguration = async () => {
    try {
      const result = await apiService.exportConfiguration();
      
      // Create and download the file
      const blob = new Blob([JSON.stringify(result.config, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `arp_config_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Failed to export configuration:', error);
      setError('Failed to export configuration');
    }
  };

  const importConfiguration = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const config = JSON.parse(text);
      
      if (!confirm('Are you sure you want to import this configuration? This will overwrite your current settings.')) {
        return;
      }

      const result = await apiService.importConfiguration(config);
      
      if (result.status === 'success') {
        await loadConfiguration();
        setError(null);
      } else {
        setError(result.message || 'Failed to import configuration');
      }
      
    } catch (error) {
      console.error('Failed to import configuration:', error);
      setError('Failed to import configuration. Please check the file format.');
    }
  };

  const createBackup = async () => {
    try {
      const result = await apiService.createConfigurationBackup();
      if (result.status === 'success') {
        await loadBackups();
        setError(null);
      } else {
        setError(result.message || 'Failed to create backup');
      }
    } catch (error) {
      console.error('Failed to create backup:', error);
      setError('Failed to create backup');
    }
  };

  const restoreBackup = async (backupName: string) => {
    if (!confirm(`Are you sure you want to restore configuration from ${backupName}? This will overwrite your current settings.`)) {
      return;
    }

    try {
      const result = await apiService.restoreConfigurationBackup(backupName);
      
      if (result.status === 'success') {
        await loadConfiguration();
        setError(null);
      } else {
        setError(result.message || 'Failed to restore backup');
      }
      
    } catch (error) {
      console.error('Failed to restore backup:', error);
      setError('Failed to restore backup');
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

  const exportRegistry = async () => {
    try {
      const result = await apiService.exportRegistry();
      
      // Create and download the file
      const blob = new Blob([JSON.stringify(result.entries, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `arp_registry_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Failed to export registry:', error);
      setError('Failed to export registry');
    }
  };

  const importRegistry = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      
      if (!confirm('Are you sure you want to import this registry? This will replace all current entries.')) {
        return;
      }

      const result = await apiService.importRegistry(data);
      
      if (result.status === 'success') {
        await loadRegistry();
        setError(null);
      } else {
        setError(result.message || 'Failed to import registry');
      }
      
    } catch (error) {
      console.error('Failed to import registry:', error);
      setError('Failed to import registry. Please check the file format.');
    }
  };

  const addRegistryEntry = async () => {
    if (!newEntry.ip || !newEntry.mac) {
      setError('Please enter both IP and MAC addresses');
      return;
    }

    try {
      const result = await apiService.addRegistryEntry(newEntry.ip, newEntry.mac);
      
      if (result.status === 'success' || result.status === 'info') {
        setNewEntry({ ip: '', mac: '' });
        await loadRegistry();
        setError(null);
      } else {
        setError(result.message || 'Failed to add entry');
      }
      
    } catch (error) {
      console.error('Failed to add registry entry:', error);
      setError('Failed to add registry entry');
    }
  };

  const removeRegistryEntry = async (ip: string) => {
    if (!confirm(`Are you sure you want to remove ${ip} from the registry?`)) {
      return;
    }

    try {
      const result = await apiService.removeRegistryEntry(ip);
      
      if (result.status === 'success' || result.status === 'info') {
        await loadRegistry();
        setError(null);
      } else {
        setError(result.message || 'Failed to remove entry');
      }
      
    } catch (error) {
      console.error('Failed to remove registry entry:', error);
      setError('Failed to remove registry entry');
    }
  };

  const resetRegistry = async () => {
    if (!confirm('Are you sure you want to reset the registry? This will remove all entries.')) {
      return;
    }

    try {
      const result = await apiService.resetRegistry();
      
      if (result.status === 'success') {
        await loadRegistry();
        setError(null);
      } else {
        setError(result.message || 'Failed to reset registry');
      }
      
    } catch (error) {
      console.error('Failed to reset registry:', error);
      setError('Failed to reset registry');
    }
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

        {/* Validation Results */}
        {validationResult && (
          <div className={`p-4 rounded-lg mb-8 ${validationResult.valid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            <h3 className="font-semibold mb-2">
              Configuration Validation: {validationResult.valid ? 'Valid' : 'Invalid'}
            </h3>
            {validationResult.errors.length > 0 && (
              <div className="mb-2">
                <strong>Errors:</strong>
                <ul className="list-disc list-inside ml-4">
                  {validationResult.errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
            {validationResult.warnings.length > 0 && (
              <div>
                <strong>Warnings:</strong>
                <ul className="list-disc list-inside ml-4">
                  {validationResult.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Quick Actions */}
        <div className="bg-white/80 border border-blue-200 rounded-xl shadow-xl p-6 mb-8">
          <div className="h-2 bg-gradient-to-r from-green-400 to-blue-400 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Quick Actions</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <button
              onClick={validateConfiguration}
              className="bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold py-3 px-4 rounded-lg shadow hover:from-blue-600 hover:to-blue-700 transition-all"
            >
              Validate Config
            </button>
            
            <button
              onClick={exportConfiguration}
              className="bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold py-3 px-4 rounded-lg shadow hover:from-green-600 hover:to-green-700 transition-all"
            >
              Export Config
            </button>
            
            <label className="bg-gradient-to-r from-purple-500 to-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow hover:from-purple-600 hover:to-purple-700 transition-all cursor-pointer text-center">
              Import Config
              <input
                type="file"
                accept=".json"
                onChange={importConfiguration}
                className="hidden"
              />
            </label>
            
            <button
              onClick={createBackup}
              className="bg-gradient-to-r from-orange-500 to-orange-600 text-white font-semibold py-3 px-4 rounded-lg shadow hover:from-orange-600 hover:to-orange-700 transition-all"
            >
              Create Backup
            </button>
          </div>
        </div>

        {/* Backups Section */}
        {backups.length > 0 && (
          <div className="bg-white/80 border border-orange-200 rounded-xl shadow-xl p-6 mb-8">
            <div className="h-2 bg-gradient-to-r from-orange-400 to-yellow-400 rounded-t-lg mb-4"></div>
            <h2 className="text-2xl font-bold text-orange-700 mb-6">Configuration Backups</h2>
            
            <div className="space-y-3">
              {backups.map((backup, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div>
                    <h3 className="font-semibold text-gray-800">{backup.name}</h3>
                    <p className="text-sm text-gray-600">
                      Created: {new Date(backup.created).toLocaleString()}
                    </p>
                    <p className="text-sm text-gray-600">
                      Size: {(backup.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                  <button
                    onClick={() => restoreBackup(backup.name)}
                    className="bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold py-2 px-4 rounded-lg shadow hover:from-blue-600 hover:to-blue-700 transition-all"
                  >
                    Restore
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Advanced Toggle */}
        <div className="text-center mb-8">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="bg-gradient-to-r from-gray-500 to-gray-600 text-white font-semibold py-2 px-6 rounded-lg shadow hover:from-gray-600 hover:to-gray-700 transition-all"
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>
        </div>

        {/* Alert Configuration */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
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
                          <input
                            type="email"
                            value={email}
                            onChange={(e) => {
                              const newRecipients = [...alertConfig.emailRecipients];
                              newRecipients[index] = e.target.value;
                              setAlertConfig(prev => ({ ...prev, emailRecipients: newRecipients }));
                            }}
                            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          />
                          <button
                            onClick={() => removeEmailRecipient(index)}
                            className="text-red-600 hover:text-red-800"
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                    <button
                      onClick={addEmailRecipient}
                      className="mt-2 text-blue-600 hover:text-blue-800 text-sm"
                    >
                      + Add Email Recipient
                    </button>
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
        {showAdvanced && (
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
        )}

        {/* Registry Management */}
        <div className="bg-white/80 border border-purple-200 rounded-xl shadow-xl p-8 mb-8">
          <div className="h-2 bg-gradient-to-r from-purple-400 to-blue-400 rounded-t-lg mb-4"></div>
          <h2 className="text-2xl font-bold text-blue-700 mb-6">Registry Management</h2>
          
          <div className="space-y-6">
            {/* Terminal-like Registry Display */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Registry Entries ({Object.keys(registryEntries).length} entries)
              </label>
              
              {/* Terminal Header */}
              <div className="bg-gray-900 text-gray-300 px-4 py-2 rounded-t-lg border-b border-gray-700 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <span className="text-sm font-mono">registry.yml</span>
                  {newEntriesDetected && (
                    <span className="text-xs text-green-400 animate-pulse">● LIVE</span>
                  )}
                </div>
                <div className="flex items-center space-x-3">
                  <button
                    onClick={loadRegistry}
                    disabled={registryLoading}
                    className="text-xs text-gray-400 hover:text-green-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Refresh registry"
                  >
                    {registryLoading ? '⟳ Loading...' : '↻ Refresh'}
                  </button>
                  <div className="text-xs text-gray-400">
                    ARP Registry Terminal
                  </div>
                </div>
              </div>
              
              {/* Terminal Body */}
              <div className="bg-black text-green-400 font-mono text-sm p-4 rounded-b-lg border border-gray-700">
                <div className="mb-2 text-gray-400">
                  <span className="text-yellow-400">$</span> cat registry.yml
                </div>
                
                {/* Scrollable Registry Content */}
                <div className="max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
                  {Object.keys(registryEntries).length === 0 ? (
                    <div className="text-gray-500 italic">
                      # Registry is empty
                      # New entries will be added automatically during monitoring
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {Object.entries(registryEntries).map(([ip, mac], index) => (
                        <div key={ip} className="flex items-center justify-between group hover:bg-gray-800 px-2 py-1 rounded">
                          <div className="flex-1">
                            <span className="text-blue-400">{ip}</span>
                            <span className="text-gray-400 mx-2">:</span>
                            <span className="text-green-400">{mac}</span>
                          </div>
                          <button
                            onClick={() => removeRegistryEntry(ip)}
                            className="text-red-400 hover:text-red-300 opacity-0 group-hover:opacity-100 transition-opacity text-xs"
                            title="Remove entry"
                          >
                            [DEL]
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
                <div className="mt-2 text-gray-400">
                  <span className="text-yellow-400">$</span> echo "Registry entries: {Object.keys(registryEntries).length}"
                </div>
                <div className="mt-1 text-blue-400">
                  <span className="text-yellow-400">$</span> echo "DEBUG: Actual count: {Object.keys(registryEntries).length}, State count: {registryEntryCount}"
                </div>
                {newEntriesDetected && (
                  <div className="mt-1 text-green-400 animate-pulse">
                    <span className="text-yellow-400">$</span> echo "🆕 New entries detected!"
                  </div>
                )}
                {registryLastUpdated && (
                  <div className="mt-1 text-gray-400">
                    <span className="text-yellow-400">$</span> echo "Last updated: {registryLastUpdated.toLocaleTimeString()}"
                  </div>
                )}
              </div>
            </div>

            {/* Registry Controls */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Add New Entry */}
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Add New Entry</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-semibold text-gray-600 mb-1">IP Address</label>
                    <input
                      type="text"
                      value={newEntry.ip}
                      onChange={(e) => setNewEntry(prev => ({ ...prev, ip: e.target.value }))}
                      placeholder="192.168.1.100"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-600 mb-1">MAC Address</label>
                    <input
                      type="text"
                      value={newEntry.mac}
                      onChange={(e) => setNewEntry(prev => ({ ...prev, mac: e.target.value }))}
                      placeholder="00:11:22:33:44:55"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                    />
                  </div>
                  <button
                    onClick={addRegistryEntry}
                    className="w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-700 transition-colors text-sm"
                  >
                    Add Entry
                  </button>
                </div>
              </div>

              {/* Registry Actions */}
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Registry Actions</h3>
                <div className="space-y-3">
                  <button
                    onClick={exportRegistry}
                    className="w-full bg-green-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-green-700 transition-colors text-sm"
                  >
                    Export Registry
                  </button>
                  
                  <label className="w-full bg-purple-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-purple-700 transition-colors text-sm cursor-pointer text-center block">
                    Import Registry
                    <input
                      type="file"
                      accept=".json"
                      onChange={importRegistry}
                      className="hidden"
                    />
                  </label>
                  
                  <button
                    onClick={resetRegistry}
                    className="w-full bg-red-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-red-700 transition-colors text-sm"
                  >
                    Reset Registry
                  </button>
                </div>
              </div>
            </div>

            {/* Registry Info */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-blue-700 mb-2">Registry Information</h3>
              <div className="text-sm text-blue-600 space-y-1">
                <p>• Registry entries are automatically added during monitoring</p>
                <p>• Only valid, non-threatening ARP entries are added</p>
                <p>• Private IP ranges (192.168.x.x, 10.x.x.x, 172.16.x.x) are preferred</p>
                <p>• Registry file: <code className="bg-blue-100 px-1 rounded">data/registry.yml</code></p>
                <p>• Auto-refresh: <span className="text-green-600 font-semibold">● Active (every 5s)</span></p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
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