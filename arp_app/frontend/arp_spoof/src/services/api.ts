// API Service for ARP Guardian Frontend
// Interfaces with the FastAPI backend

export interface NetworkInterface {
  name: string;
  description?: string;
  status?: 'up' | 'down';
}

export interface MonitoringStatus {
  is_monitoring: boolean;
  current_interface: string | null;
  live_stats: {
    total_packets: number;
    arp_packets: number;
    detected_attacks: number;
    last_attack_time: string | null;
    current_interface: string | null;
    monitoring_status: string;
    // Prevention statistics
    prevention_active: boolean;
    packets_dropped: number;
    arp_entries_corrected: number;
    quarantined_ips: number;
    rate_limited_ips: number;
  };
  recent_detections_count: number;
}

export interface DetectionRecord {
  id: number;
  timestamp: string;
  src_ip: string;
  src_mac: string;
  dst_ip: string;
  threat_level: string;
  rule_detection: boolean;
  rule_reason: string;
  ml_prediction: boolean;
  ml_confidence: number;
}

export interface DetectionStats {
  total_packets: number;
  arp_packets: number;
  detected_attacks: number;
  monitoring_status: string;
  current_interface: string | null;
  recent_detections: DetectionRecord[];
}

export interface Alert {
  id: string;
  timestamp: string;
  type: 'arp_spoofing' | 'suspicious_activity' | 'system_error' | 'configuration_change';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  sourceIP?: string;
  targetIP?: string;
  status: 'new' | 'acknowledged' | 'resolved' | 'dismissed';
  acknowledgedBy?: string;
  resolvedAt?: string;
}

export interface AlertStats {
  total: number;
  new: number;
  acknowledged: number;
  resolved: number;
  dismissed: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

export interface AlertConfig {
  emailEnabled: boolean;
  emailRecipients: string[];
  webhookEnabled: boolean;
  webhookUrl: string;
  notificationCooldown: number;
  enableDesktopNotifications: boolean;
  enableSoundAlerts: boolean;
}

export interface SystemConfig {
  autoStart: boolean;
  logLevel: 'debug' | 'info' | 'warning' | 'error';
  maxLogSize: number;
  enableBackup: boolean;
  backupInterval: number;
}

export interface WebSocketMessage {
  type: 'monitoring_status' | 'attack_detected' | 'stats_update' | 'alert' | 'prevention_action';
  data?: any;
  status?: string;
  interface?: string;
  timestamp?: string;
}

class APIService {
  private baseURL: string;
  private ws: WebSocket | null = null;
  private wsReconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor() {
    // Use environment variable or default to localhost:8000
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  }

  // Generic API request helper
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}/api/v1${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // Monitoring endpoints
  async startMonitoring(interfaceName: string, config?: any): Promise<{ status: string; message: string }> {
    return this.request('/monitoring/start', {
      method: 'POST',
      body: JSON.stringify({
        interface: interfaceName,
        config: config
      })
    });
  }

  async stopMonitoring(): Promise<{ status: string; message: string }> {
    return this.request('/monitoring/stop', {
      method: 'POST'
    });
  }

  async getMonitoringStatus(): Promise<MonitoringStatus> {
    return this.request('/monitoring/status');
  }

  async getNetworkInterfaces(): Promise<{ interfaces: string[] }> {
    return this.request('/monitoring/interfaces');
  }

  // Statistics endpoints
  async getStatistics(): Promise<DetectionStats> {
    return this.request('/statistics');
  }

  async getRecentDetections(limit: number = 50): Promise<{ detections: DetectionRecord[]; count: number }> {
    try {
      const response = await this.request<any>(`/detections?limit=${limit}`);
      
      // Handle case where backend returns error in response body
      if (response.error) {
        console.warn('Backend returned error for detections:', response.error);
        return { detections: [], count: 0 };
      }
      
      // Ensure we have the expected structure
      if (!response.detections || !Array.isArray(response.detections)) {
        console.warn('Invalid detections response structure:', response);
        return { detections: [], count: 0 };
      }
      
      return {
        detections: response.detections,
        count: response.count || response.detections.length
      };
    } catch (error) {
      console.error('Error fetching recent detections:', error);
      return { detections: [], count: 0 };
    }
  }

  // Configuration endpoints
  async getConfiguration(): Promise<{
    alerts: AlertConfig;
    system: SystemConfig;
  }> {
    const response = await this.request<any>('/config');
    
    // Convert backend snake_case to frontend camelCase
    return {
      alerts: {
        emailEnabled: response.alerts?.email_enabled || false,
        emailRecipients: response.alerts?.email_recipients || [],
        webhookEnabled: response.alerts?.webhook_enabled || false,
        webhookUrl: response.alerts?.webhook_url || '',
        notificationCooldown: response.alerts?.notification_cooldown || 300,
        enableDesktopNotifications: response.alerts?.enable_desktop_notifications !== false,
        enableSoundAlerts: response.alerts?.enable_sound_alerts !== false
      },
      system: {
        autoStart: response.system?.auto_start || false,
        logLevel: response.system?.log_level || 'info',
        maxLogSize: response.system?.max_log_size || 100,
        enableBackup: response.system?.enable_backup !== false,
        backupInterval: response.system?.backup_interval || 24
      }
    };
  }

  async updateConfiguration(config: {
    alerts?: Partial<AlertConfig>;
  }): Promise<{ status: string; message: string }> {
    // Convert frontend camelCase to backend snake_case
    const backendConfig: any = {};
    
    if (config.alerts) {
      backendConfig.alerts = {
        email_enabled: config.alerts.emailEnabled,
        email_recipients: config.alerts.emailRecipients,
        webhook_enabled: config.alerts.webhookEnabled,
        webhook_url: config.alerts.webhookUrl,
        notification_cooldown: config.alerts.notificationCooldown,
        enable_desktop_notifications: config.alerts.enableDesktopNotifications,
        enable_sound_alerts: config.alerts.enableSoundAlerts
      };
    }
    
    return this.request('/config', {
      method: 'PUT',
      body: JSON.stringify(backendConfig)
    });
  }

  async resetConfiguration(): Promise<{ status: string; message: string }> {
    return this.request('/config/reset', {
      method: 'POST'
    });
  }

  async exportConfiguration(): Promise<{
    status: string;
    config: any;
    exported_at: string;
    version: string;
  }> {
    return this.request('/config/export');
  }

  async importConfiguration(config: any): Promise<{ status: string; message: string }> {
    return this.request('/config/import', {
      method: 'POST',
      body: JSON.stringify({ config })
    });
  }

  async validateConfiguration(): Promise<{
    status: string;
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    return this.request('/config/validate');
  }

  async createConfigurationBackup(): Promise<{
    status: string;
    message: string;
    backup_path: string;
  }> {
    return this.request('/config/backup');
  }

  async listConfigurationBackups(): Promise<{
    status: string;
    backups: Array<{
      name: string;
      path: string;
      created: string;
      size: number;
    }>;
  }> {
    return this.request('/config/backups');
  }

  async restoreConfigurationBackup(backupName: string): Promise<{ status: string; message: string }> {
    return this.request(`/config/restore/${encodeURIComponent(backupName)}`, {
      method: 'POST'
    });
  }

  async getConfigurationSchema(): Promise<any> {
    return this.request('/config/schema');
  }

  // Alert endpoints
  async getAlerts(limit: number = 50): Promise<{ alerts: Alert[]; count: number }> {
    return this.request(`/alerts?limit=${limit}`);
  }

  async getAlertStats(): Promise<AlertStats> {
    return this.request('/alerts/stats');
  }

  async clearAlerts(): Promise<{ status: string; message: string }> {
    return this.request('/alerts', {
      method: 'DELETE'
    });
  }

  // WebSocket connection
  connectWebSocket(onMessage: (message: WebSocketMessage) => void): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    const wsUrl = this.baseURL.replace('http', 'ws') + '/ws';
    console.log('Connecting to WebSocket:', wsUrl);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected successfully');
      this.wsReconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      console.log('WebSocket raw message received:', event.data);
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        console.log('WebSocket parsed message:', message);
        onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      this.attemptReconnect(onMessage);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private attemptReconnect(onMessage: (message: WebSocketMessage) => void): void {
    if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
      this.wsReconnectAttempts++;
      console.log(`Attempting to reconnect WebSocket (${this.wsReconnectAttempts}/${this.maxReconnectAttempts})`);
      
      // Clear the current WebSocket reference
      this.ws = null;
      
      setTimeout(() => {
        this.connectWebSocket(onMessage);
      }, this.reconnectDelay * this.wsReconnectAttempts);
    } else {
      console.error('Max WebSocket reconnection attempts reached');
    }
  }

  disconnectWebSocket(): void {
    if (this.ws) {
      console.log('Disconnecting WebSocket...');
      this.ws.close();
      this.ws = null;
      this.wsReconnectAttempts = 0;
    }
  }

  // Health check
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    services: {
      arp_detection: boolean;
      config: boolean;
      alerts: boolean;
      websocket: boolean;
    };
  }> {
    return this.request('/health', { method: 'GET' });
  }

  // Utility method to check if backend is available
  async isBackendAvailable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('Backend not available:', error);
      return false;
    }
  }

  // Simple ping to test basic connectivity
  async ping(): Promise<{ status: string; message: string; timestamp: string }> {
    try {
      const response = await fetch(`${this.baseURL}/ping`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Ping failed:', error);
      throw error;
    }
  }

  // Test detections endpoint - COMMENTED OUT: Automatic registry addition is now working properly
  // async testDetectionsEndpoint(): Promise<{ status: string; message: string; detections_count: number }> {
  //   try {
  //     const response = await this.request<any>('/test-detections');
  //     return response;
  //   } catch (error) {
  //     console.error('Test detections endpoint failed:', error);
  //     return {
  //       status: 'failed',
  //       message: 'Failed to test detections endpoint',
  //       detections_count: 0
  //     };
  //   }
  // }

  // Registry management endpoints
  async getRegistryEntries(): Promise<{
    status: string;
    entries: Record<string, string>;
    count: number;
  }> {
    return this.request('/registry/direct');
  }

  async exportRegistry(): Promise<{
    status: string;
    entries: Record<string, string>;
    exported_at: string;
    count: number;
  }> {
    return this.request('/registry/export');
  }

  async importRegistry(entries: Record<string, string>): Promise<{ status: string; message: string }> {
    return this.request('/registry/import', {
      method: 'POST',
      body: JSON.stringify({ entries })
    });
  }

  async addRegistryEntry(ip: string, mac: string): Promise<{ status: string; message: string }> {
    return this.request('/registry/add', {
      method: 'POST',
      body: JSON.stringify({ ip, mac })
    });
  }

  async removeRegistryEntry(ip: string): Promise<{ status: string; message: string }> {
    return this.request(`/registry/remove/${encodeURIComponent(ip)}`, {
      method: 'DELETE'
    });
  }

  async resetRegistry(): Promise<{ status: string; message: string }> {
    return this.request('/registry/reset', {
      method: 'POST'
    });
  }

  // Prevention endpoints
  async getPreventionStats(): Promise<{
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
  }> {
    return this.request('/prevention/stats');
  }

  async getQuarantineList(): Promise<{
    quarantine_list: Array<{
      ip: string;
      mac: string;
      reason: string;
      quarantined_at: string;
      expires_at: string;
      attempts: number;
    }>;
    count: number;
  }> {
    return this.request('/prevention/quarantine');
  }

  async getRateLimits(): Promise<{
    rate_limits: Array<{
      ip: string;
      mac: string;
      first_seen: string;
      last_seen: string;
      packet_count: number;
      blocked_until: string | null;
    }>;
    count: number;
  }> {
    return this.request('/prevention/rate-limits');
  }

  async addLegitimateEntry(ip: string, mac: string): Promise<{ status: string; message: string }> {
    return this.request('/prevention/legitimate', {
      method: 'POST',
      body: JSON.stringify({ ip, mac })
    });
  }

  async removeQuarantineEntry(ip: string): Promise<{ status: string; message: string }> {
    return this.request(`/prevention/quarantine/${encodeURIComponent(ip)}`, {
      method: 'DELETE'
    });
  }

  async clearPreventionData(): Promise<{ status: string; message: string }> {
    return this.request('/prevention/clear', {
      method: 'POST'
    });
  }

  async getArpTable(): Promise<{
    arp_table: Array<{
      ip: string;
      mac: string;
      type: string;
      interface: string | null;
      is_legitimate: boolean;
    }>;
    count: number;
  }> {
    return this.request('/prevention/arp-table');
  }

  async testPreventionAction(ip: string, mac: string, threatLevel: string = 'HIGH'): Promise<{
    success: boolean;
    message: string;
    action?: string;
    reason?: string;
  }> {
    return this.request('/prevention/test', {
      method: 'POST',
      body: JSON.stringify({ ip, mac, threat_level: threatLevel })
    });
  }
}

// Export singleton instance
export const apiService = new APIService();
export default apiService; 