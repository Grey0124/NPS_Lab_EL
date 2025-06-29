import notificationService from './notificationService';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface AlertNotification {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: string;
  sourceIP?: string;
  targetIP?: string;
  timestamp: string;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private url: string;
  private isConnecting = false;
  private messageHandlers: Map<string, (data: any) => void> = new Map();

  constructor() {
    // Use the same base URL as the API service
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const rawHost = import.meta.env.VITE_API_URL || 'localhost:8000';
    const host = rawHost.replace(/^https?:\/\//, '');
    this.url = `${protocol}//${host}/ws`;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;

    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnecting = false;
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
      };

    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      this.isConnecting = false;
      this.handleReconnect();
    }
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(message: WebSocketMessage): void {
    console.log('Received WebSocket message:', message);

    // Handle different message types
    switch (message.type) {
      case 'new_alert':
        this.handleNewAlert(message.data);
        break;
      case 'attack_detected':
        this.handleAttackDetected(message.data);
        break;
      case 'stats_update':
        this.handleStatsUpdate(message.data);
        break;
      case 'system_status':
        this.handleSystemStatus(message.data);
        break;
      default:
        // Call registered handlers
        const handler = this.messageHandlers.get(message.type);
        if (handler) {
          handler(message.data);
        }
    }
  }

  /**
   * Handle new alert notifications
   */
  private handleNewAlert(alert: AlertNotification): void {
    console.log('New alert received:', alert);
    
    // Show desktop notification if enabled
    if (notificationService.isAvailable()) {
      notificationService.showSecurityAlert(alert);
    }

    // Emit event for components to listen to
    window.dispatchEvent(new CustomEvent('newAlert', { detail: alert }));
  }

  /**
   * Handle attack detection notifications
   */
  private handleAttackDetected(detection: any): void {
    console.log('Attack detected:', detection);
    
    // Emit event for components to listen to
    window.dispatchEvent(new CustomEvent('attackDetected', { detail: detection }));
  }

  /**
   * Handle stats updates
   */
  private handleStatsUpdate(stats: any): void {
    console.log('Stats update received:', stats);
    window.dispatchEvent(new CustomEvent('statsUpdate', { detail: stats }));
  }

  /**
   * Handle system status updates
   */
  private handleSystemStatus(status: any): void {
    console.log('System status update received:', status);
    window.dispatchEvent(new CustomEvent('systemStatus', { detail: status }));
  }

  /**
   * Send message to WebSocket server
   */
  send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }

  /**
   * Register a message handler
   */
  onMessage(type: string, handler: (data: any) => void): void {
    this.messageHandlers.set(type, handler);
  }

  /**
   * Remove a message handler
   */
  offMessage(type: string): void {
    this.messageHandlers.delete(type);
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnecting = false;
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection status
   */
  getStatus(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }
}

// Create a singleton instance
const websocketService = new WebSocketService();

export default websocketService; 