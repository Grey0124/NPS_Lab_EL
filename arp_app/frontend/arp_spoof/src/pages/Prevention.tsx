import React, { useState, useEffect } from 'react';
import apiService from '../services/api';

interface PreventionStats {
  is_active: boolean;
  current_interface: string | null;
  total_packets_dropped: number;
  total_arp_entries_corrected: number;
  total_quarantined_ips: number;
  total_rate_limited_ips: number;
  last_prevention_time: string | null;
  prevention_duration: number;
  quarantine_count: number;
  rate_limit_count: number;
  arp_table_size: number;
  legitimate_entries: number;
}

interface QuarantineEntry {
  ip: string;
  mac: string;
  reason: string;
  quarantined_at: string;
  expires_at: string;
  attempts: number;
}

interface RateLimitEntry {
  ip: string;
  mac: string;
  first_seen: string;
  last_seen: string;
  packet_count: number;
  blocked_until: string | null;
}

interface ArpTableEntry {
  ip: string;
  mac: string;
  type: string;
  interface: string | null;
  is_legitimate: boolean;
}

const Prevention: React.FC = () => {
  const [preventionStats, setPreventionStats] = useState<PreventionStats | null>(null);
  const [quarantineList, setQuarantineList] = useState<QuarantineEntry[]>([]);
  const [rateLimits, setRateLimits] = useState<RateLimitEntry[]>([]);
  const [arpTable, setArpTable] = useState<ArpTableEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newLegitimateIP, setNewLegitimateIP] = useState('');
  const [newLegitimateMAC, setNewLegitimateMAC] = useState('');
  const [testIP, setTestIP] = useState('');
  const [testMAC, setTestMAC] = useState('');
  const [testThreatLevel, setTestThreatLevel] = useState('HIGH');
  const [arpTableSearch, setArpTableSearch] = useState('');
  const [quarantineSearch, setQuarantineSearch] = useState('');

  useEffect(() => {
    loadPreventionData();
    
    // Connect to WebSocket for real-time updates
    apiService.connectWebSocket((message) => {
      if (message.type === 'stats_update' && message.data) {
        // Update prevention stats directly from WebSocket data
        // This ensures real-time updates without API calls
        if (preventionStats) {
          setPreventionStats(prev => {
            if (!prev) return prev;
            
            // Update prevention stats from WebSocket data
            // Map backend field names to frontend interface names
            const updatedStats = {
              ...prev,
              total_packets_dropped: message.data.packets_dropped !== undefined ? message.data.packets_dropped : prev.total_packets_dropped,
              total_quarantined_ips: message.data.quarantined_ips !== undefined ? message.data.quarantined_ips : prev.total_quarantined_ips,
              total_arp_entries_corrected: message.data.arp_entries_corrected !== undefined ? message.data.arp_entries_corrected : prev.total_arp_entries_corrected,
              total_rate_limited_ips: message.data.rate_limited_ips !== undefined ? message.data.rate_limited_ips : prev.total_rate_limited_ips,
              is_active: message.data.prevention_active !== undefined ? message.data.prevention_active : prev.is_active
            };
            
            console.log('Updated prevention stats from WebSocket:', updatedStats);
            return updatedStats;
          });
        }
      }
      if (message.type === 'prevention_action') {
        // Reload prevention data when prevention actions occur
        loadPreventionData();
      }
    });
    
    return () => {
      apiService.disconnectWebSocket();
    };
  }, [preventionStats]);

  // Keyboard shortcuts for terminal
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ctrl+F to focus search
      if (event.ctrlKey && event.key === 'f') {
        event.preventDefault();
        // Try to focus ARP table search first, then quarantine search
        const arpSearchInput = document.querySelector('input[placeholder*="arp -a"]') as HTMLInputElement;
        const quarantineSearchInput = document.querySelector('input[placeholder*="quarantine list"]') as HTMLInputElement;
        
        if (arpSearchInput && document.activeElement !== arpSearchInput) {
          arpSearchInput.focus();
        } else if (quarantineSearchInput && document.activeElement !== quarantineSearchInput) {
          quarantineSearchInput.focus();
        }
      }
      // Escape to clear search
      if (event.key === 'Escape') {
        setArpTableSearch('');
        setQuarantineSearch('');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const loadPreventionData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [stats, quarantine, rateLimits, arpTableData] = await Promise.all([
        apiService.getPreventionStats(),
        apiService.getQuarantineList(),
        apiService.getRateLimits(),
        apiService.getArpTable()
      ]);

      setPreventionStats(stats);
      setQuarantineList(quarantine.quarantine_list);
      setRateLimits(rateLimits.rate_limits);
      setArpTable(arpTableData.arp_table);
    } catch (error) {
      console.error('Failed to load prevention data:', error);
      setError('Failed to load prevention data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddLegitimateEntry = async () => {
    if (!newLegitimateIP || !newLegitimateMAC) {
      setError('Please enter both IP and MAC address');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const result = await apiService.addLegitimateEntry(newLegitimateIP, newLegitimateMAC);
      
      if (result.status === 'success') {
        setNewLegitimateIP('');
        setNewLegitimateMAC('');
        await loadPreventionData();
      } else {
        setError(result.message);
      }
    } catch (error) {
      console.error('Failed to add legitimate entry:', error);
      setError('Failed to add legitimate entry');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRemoveQuarantine = async (ip: string) => {
    try {
      setIsLoading(true);
      setError(null);

      const result = await apiService.removeQuarantineEntry(ip);
      
      if (result.status === 'success') {
        await loadPreventionData();
      } else {
        setError(result.message);
      }
    } catch (error) {
      console.error('Failed to remove quarantine entry:', error);
      setError('Failed to remove quarantine entry');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearPreventionData = async () => {
    if (!window.confirm('Are you sure you want to clear all prevention data? This action cannot be undone.')) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const result = await apiService.clearPreventionData();
      
      if (result.status === 'success') {
        await loadPreventionData();
      } else {
        setError(result.message);
      }
    } catch (error) {
      console.error('Failed to clear prevention data:', error);
      setError('Failed to clear prevention data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTestPrevention = async () => {
    if (!testIP || !testMAC) {
      setError('Please enter both IP and MAC address for testing');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const result = await apiService.testPreventionAction(testIP, testMAC, testThreatLevel);
      
      if (result.success) {
        alert(`Test completed: ${result.action} - ${result.reason}`);
        await loadPreventionData();
      } else {
        setError(result.message);
      }
    } catch (error) {
      console.error('Failed to test prevention:', error);
      setError('Failed to test prevention action');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };

  const formatDateTime = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  // Filter ARP table based on search
  const filteredArpTable = arpTable.filter(entry => 
    entry.ip.toLowerCase().includes(arpTableSearch.toLowerCase()) ||
    entry.mac.toLowerCase().includes(arpTableSearch.toLowerCase()) ||
    entry.type.toLowerCase().includes(arpTableSearch.toLowerCase()) ||
    (entry.interface && entry.interface.toLowerCase().includes(arpTableSearch.toLowerCase()))
  );

  // Group ARP table entries by interface
  const groupedArpTable = filteredArpTable.reduce((groups, entry) => {
    const interfaceName = entry.interface || 'Unknown';
    if (!groups[interfaceName]) {
      groups[interfaceName] = [];
    }
    groups[interfaceName].push(entry);
    return groups;
  }, {} as Record<string, typeof filteredArpTable>);

  // Filter quarantine list based on search
  const filteredQuarantineList = quarantineList.filter(entry => 
    entry.ip.toLowerCase().includes(quarantineSearch.toLowerCase()) ||
    entry.mac.toLowerCase().includes(quarantineSearch.toLowerCase()) ||
    entry.reason.toLowerCase().includes(quarantineSearch.toLowerCase())
  );

  if (isLoading && !preventionStats) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">ARP Prevention</h1>
        <button
          onClick={loadPreventionData}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Prevention Statistics */}
      {preventionStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Status</h3>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${preventionStats.is_active ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {preventionStats.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            {preventionStats.current_interface && (
              <p className="text-sm text-gray-600 mt-1">
                Interface: {preventionStats.current_interface}
              </p>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Packets Dropped</h3>
            <p className="text-3xl font-bold text-red-600">{preventionStats.total_packets_dropped}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">ARP Entries Corrected</h3>
            <p className="text-3xl font-bold text-blue-600">{preventionStats.total_arp_entries_corrected}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Quarantined IPs</h3>
            <p className="text-3xl font-bold text-orange-600">{preventionStats.total_quarantined_ips}</p>
          </div>
        </div>
      )}

      {/* Add Legitimate Entry */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Add Legitimate Entry</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <input
            type="text"
            placeholder="IP Address"
            value={newLegitimateIP}
            onChange={(e) => setNewLegitimateIP(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="text"
            placeholder="MAC Address"
            value={newLegitimateMAC}
            onChange={(e) => setNewLegitimateMAC(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleAddLegitimateEntry}
            disabled={isLoading || !newLegitimateIP || !newLegitimateMAC}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            Add Entry
          </button>
        </div>
      </div>

      {/* Test Prevention */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Test Prevention</h2>
        <p className="text-gray-600 mb-4">
          Test the prevention system by simulating a threat with custom IP and MAC addresses.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <input
            type="text"
            placeholder="Test IP Address"
            value={testIP}
            onChange={(e) => setTestIP(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="text"
            placeholder="Test MAC Address"
            value={testMAC}
            onChange={(e) => setTestMAC(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <select
            value={testThreatLevel}
            onChange={(e) => setTestThreatLevel(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="LOW">Low Threat</option>
            <option value="MEDIUM">Medium Threat</option>
            <option value="HIGH">High Threat</option>
          </select>
          <button
            onClick={handleTestPrevention}
            disabled={isLoading || !testIP || !testMAC}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
          >
            Test Prevention
          </button>
        </div>
      </div>

      {/* Quarantine List - Terminal Style */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Quarantine List</h2>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">{filteredQuarantineList.length} entries</span>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
          </div>
        </div>
        
        {filteredQuarantineList.length === 0 ? (
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-red-400">●</span>
              <span className="text-yellow-400">●</span>
              <span className="text-green-400">●</span>
              <span className="text-gray-400">Quarantine Terminal</span>
            </div>
            <div className="border-t border-gray-700 pt-2">
              <span className="text-gray-400">$</span> <span className="text-gray-300">quarantine list</span>
              <br />
              <span className="text-gray-500">No quarantined IPs found</span>
            </div>
          </div>
        ) : (
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
            {/* Terminal Header */}
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-red-400">●</span>
              <span className="text-yellow-400">●</span>
              <span className="text-green-400">●</span>
              <span className="text-gray-400">Quarantine Terminal</span>
            </div>
            
            {/* Search Input */}
            <div className="mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-gray-400">$</span>
                <input
                  type="text"
                  placeholder="grep -i 'search_term' | quarantine list"
                  value={quarantineSearch}
                  onChange={(e) => setQuarantineSearch(e.target.value)}
                  className="flex-1 bg-transparent text-green-400 border-none outline-none font-mono text-sm placeholder-gray-600"
                />
                {quarantineSearch && (
                  <button
                    onClick={() => setQuarantineSearch('')}
                    className="text-gray-400 hover:text-red-400 text-sm"
                    title="Clear search"
                  >
                    ✕
                  </button>
                )}
              </div>
              <div className="text-xs text-gray-600 mt-1 ml-6">
                <span className="text-gray-500">Ctrl+F: Focus search | Esc: Clear search</span>
              </div>
            </div>
            
            {/* Terminal Content with Scroll */}
            <div className="border-t border-gray-700 pt-2">
              <div className="mb-2">
                <span className="text-gray-400">$</span> <span className="text-gray-300">quarantine list</span>
                {quarantineSearch && (
                  <span className="text-gray-300"> | grep -i "{quarantineSearch}"</span>
                )}
              </div>
              
              {/* Scrollable Quarantine List */}
              <div className="max-h-96 overflow-y-auto terminal-scrollbar">
                <div className="space-y-1">
                  {filteredQuarantineList.map((entry, index) => (
                    <div key={`quarantine-${entry.ip}-${index}`} className="flex items-center space-x-4 py-1">
                      {/* IP Address */}
                      <span className="text-red-400 w-32 truncate" title={entry.ip}>
                        {entry.ip}
                      </span>
                      
                      {/* MAC Address */}
                      <span className="text-green-400 w-40 truncate" title={entry.mac}>
                        {entry.mac}
                      </span>
                      
                      {/* Reason */}
                      <span className="text-yellow-400 w-48 truncate" title={entry.reason}>
                        {entry.reason}
                      </span>
                      
                      {/* Quarantined At */}
                      <span className="text-blue-400 w-32 truncate" title={formatDateTime(entry.quarantined_at)}>
                        {new Date(entry.quarantined_at).toLocaleDateString()}
                      </span>
                      
                      {/* Expires At */}
                      <span className="text-cyan-400 w-32 truncate" title={formatDateTime(entry.expires_at)}>
                        {new Date(entry.expires_at).toLocaleDateString()}
                      </span>
                      
                      {/* Attempts Badge */}
                      <span className={`px-2 py-0.5 text-xs rounded ${
                        entry.attempts > 5 
                          ? 'bg-red-900 text-red-300 border border-red-700' 
                          : 'bg-yellow-900 text-yellow-300 border border-yellow-700'
                      }`}>
                        {entry.attempts} attempts
                      </span>
                      
                      {/* Remove Button */}
                      <button
                        onClick={() => handleRemoveQuarantine(entry.ip)}
                        disabled={isLoading}
                        className="text-red-400 hover:text-red-300 disabled:opacity-50 text-sm"
                        title="Remove from quarantine"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Terminal Footer */}
              <div className="mt-2 pt-2 border-t border-gray-700">
                <span className="text-gray-400">$</span> <span className="text-gray-300">echo "Total quarantined: {quarantineList.length}"</span>
                <br />
                <span className="text-gray-500">Total quarantined: {quarantineList.length}</span>
                {quarantineSearch && (
                  <>
                    <br />
                    <span className="text-gray-400">$</span> <span className="text-gray-300">echo "Filtered quarantined: {filteredQuarantineList.length}"</span>
                    <br />
                    <span className="text-gray-500">Filtered quarantined: {filteredQuarantineList.length}</span>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Rate Limits */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Rate Limits</h2>
          <span className="text-sm text-gray-600">{rateLimits.length} entries</span>
        </div>
        
        {rateLimits.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No rate-limited IPs</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">IP Address</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAC Address</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Packet Count</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">First Seen</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Seen</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Blocked Until</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {rateLimits.map((entry, index) => (
                  <tr key={`rate-limit-${entry.ip}-${index}`}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{entry.ip}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{entry.mac}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{entry.packet_count}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatDateTime(entry.first_seen)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatDateTime(entry.last_seen)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {entry.blocked_until ? formatDateTime(entry.blocked_until) : 'Not blocked'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ARP Table - Terminal Style */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-900">ARP Table</h2>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600">{filteredArpTable.length} entries</span>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
          </div>
        </div>
        
        {filteredArpTable.length === 0 ? (
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-red-400">●</span>
              <span className="text-yellow-400">●</span>
              <span className="text-green-400">●</span>
              <span className="text-gray-400">ARP Table Terminal</span>
            </div>
            <div className="border-t border-gray-700 pt-2">
              <span className="text-gray-400">$</span> <span className="text-gray-300">arp -a</span>
              <br />
              <span className="text-gray-500">No ARP table entries found</span>
            </div>
          </div>
        ) : (
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
            {/* Terminal Header */}
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-red-400">●</span>
              <span className="text-yellow-400">●</span>
              <span className="text-green-400">●</span>
              <span className="text-gray-400">ARP Table Terminal</span>
            </div>
            
            {/* Search Input */}
            <div className="mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-gray-400">$</span>
                <input
                  type="text"
                  placeholder="grep -i 'search_term' | arp -a"
                  value={arpTableSearch}
                  onChange={(e) => setArpTableSearch(e.target.value)}
                  className="flex-1 bg-transparent text-green-400 border-none outline-none font-mono text-sm placeholder-gray-600"
                />
                {arpTableSearch && (
                  <button
                    onClick={() => setArpTableSearch('')}
                    className="text-gray-400 hover:text-red-400 text-sm"
                    title="Clear search"
                  >
                    ✕
                  </button>
                )}
              </div>
              <div className="text-xs text-gray-600 mt-1 ml-6">
                <span className="text-gray-500">Ctrl+F: Focus search | Esc: Clear search</span>
              </div>
            </div>
            
            {/* Terminal Content with Scroll */}
            <div className="border-t border-gray-700 pt-2">
              <div className="mb-2">
                <span className="text-gray-400">$</span> <span className="text-gray-300">arp -a</span>
                {arpTableSearch && (
                  <span className="text-gray-300"> | grep -i "{arpTableSearch}"</span>
                )}
              </div>
              
              {/* Scrollable ARP Table */}
              <div className="max-h-96 overflow-y-auto terminal-scrollbar">
                <div className="space-y-1">
                  {Object.entries(groupedArpTable).map(([interfaceName, entries]) => (
                    <div key={interfaceName}>
                      {/* Interface Header */}
                      <div className="text-cyan-400 font-semibold py-1 border-b border-gray-700">
                        Interface: {interfaceName}
                      </div>
                      
                      {/* Interface Entries */}
                      {entries.map((entry, index) => (
                        <div key={`${entry.ip}-${entry.mac}-${index}`} className="flex items-center space-x-4 py-1 ml-4">
                          {/* IP Address */}
                          <span className="text-blue-400 w-32 truncate" title={entry.ip}>
                            {entry.ip}
                          </span>
                          
                          {/* MAC Address */}
                          <span className="text-green-400 w-40 truncate" title={entry.mac}>
                            {entry.mac}
                          </span>
                          
                          {/* Type Badge */}
                          <span className={`px-2 py-0.5 text-xs rounded ${
                            entry.type === 'static' 
                              ? 'bg-blue-900 text-blue-300 border border-blue-700' 
                              : 'bg-green-900 text-green-300 border border-green-700'
                          }`}>
                            {entry.type}
                          </span>
                          
                          {/* Status Badge */}
                          <span className={`px-2 py-0.5 text-xs rounded ${
                            entry.is_legitimate 
                              ? 'bg-green-900 text-green-300 border border-green-700' 
                              : 'bg-yellow-900 text-yellow-300 border border-yellow-700'
                          }`}>
                            {entry.is_legitimate ? '✓ Legitimate' : '? Unknown'}
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Terminal Footer */}
            <div className="mt-2 pt-2 border-t border-gray-700">
              <span className="text-gray-400">$</span> <span className="text-gray-300">echo "Total entries: {arpTable.length}"</span>
              <br />
              <span className="text-gray-500">Total entries: {arpTable.length}</span>
              {arpTableSearch && (
                <>
                  <br />
                  <span className="text-gray-400">$</span> <span className="text-gray-300">echo "Filtered entries: {filteredArpTable.length}"</span>
                  <br />
                  <span className="text-gray-500">Filtered entries: {filteredArpTable.length}</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Clear Prevention Data */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Danger Zone</h2>
        <p className="text-gray-600 mb-4">
          Clear all prevention data including quarantine list, rate limits, and legitimate entries.
        </p>
        <button
          onClick={handleClearPreventionData}
          disabled={isLoading}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
        >
          Clear All Prevention Data
        </button>
      </div>
    </div>
  );
};

export default Prevention; 