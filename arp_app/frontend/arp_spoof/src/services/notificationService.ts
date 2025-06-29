export interface NotificationOptions {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  tag?: string;
  requireInteraction?: boolean;
  silent?: boolean;
  data?: any;
}

export interface NotificationAction {
  action: string;
  title: string;
  icon?: string;
}

class NotificationService {
  private permission: NotificationPermission = 'default';
  private isSupported: boolean;

  constructor() {
    this.isSupported = 'Notification' in window;
    if (this.isSupported) {
      this.permission = Notification.permission;
    }
  }

  /**
   * Request permission to show notifications
   */
  async requestPermission(): Promise<boolean> {
    if (!this.isSupported) {
      console.warn('Notifications are not supported in this browser');
      return false;
    }

    if (this.permission === 'granted') {
      return true;
    }

    if (this.permission === 'denied') {
      console.warn('Notification permission has been denied');
      return false;
    }

    try {
      const permission = await Notification.requestPermission();
      this.permission = permission;
      return permission === 'granted';
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      return false;
    }
  }

  /**
   * Check if notifications are supported and permitted
   */
  isAvailable(): boolean {
    return this.isSupported && this.permission === 'granted';
  }

  /**
   * Show a desktop notification with enhanced browser compatibility
   */
  async showNotification(options: NotificationOptions): Promise<Notification | null> {
    if (!this.isAvailable()) {
      console.warn('Notifications not available or permission not granted');
      return null;
    }

    try {
      // Create a simple security icon as data URL
      const securityIcon = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMjAgN1YxMkMyMCAxNi40MTU3IDE2LjQxNTcgMjAgMTIgMjBDNy41ODQzNCAyMCA0IDE2LjQxNTcgNCAxMlY3TDEyIDJaIiBmaWxsPSIjM0I4MkZGIi8+CjxwYXRoIGQ9Ik05IDEyTDEyIDE1TDE1IDEyIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K';

      // Enhanced notification options for better browser compatibility
      const notificationOptions = {
        body: options.body,
        icon: options.icon || securityIcon,
        badge: options.badge || securityIcon,
        tag: options.tag || 'arp-guardian-notification',
        requireInteraction: options.requireInteraction || false,
        silent: options.silent || false,
        data: options.data || {},
        // Add additional options for better desktop notification behavior
        dir: 'auto' as NotificationDirection,
        lang: 'en-US',
        renotify: true,
        vibrate: options.silent ? [] : [200, 100, 200]
      };

      const notification = new Notification(options.title, notificationOptions);

      // Auto-close after 10 seconds if not requiring interaction
      if (!options.requireInteraction) {
        setTimeout(() => {
          if (notification) {
            notification.close();
          }
        }, 10000);
      }

      // Add click handler to focus the window
      notification.onclick = () => {
        window.focus();
        notification.close();
      };

      return notification;
    } catch (error) {
      console.error('Error showing notification:', error);
      return null;
    }
  }

  /**
   * Show a security alert notification
   */
  async showSecurityAlert(alert: any): Promise<Notification | null> {
    const severityEmoji = {
      critical: '🚨',
      high: '⚠️',
      medium: '⚡',
      low: 'ℹ️'
    };

    const options: NotificationOptions = {
      title: `${severityEmoji[alert.severity as keyof typeof severityEmoji] || '🚨'} Security Alert: ${alert.title}`,
      body: alert.description,
      tag: `alert-${alert.id}`,
      requireInteraction: alert.severity === 'critical' || alert.severity === 'high',
      silent: false,
      data: { alertId: alert.id, alert }
    };

    return this.showNotification(options);
  }

  /**
   * Show a test notification
   */
  async showTestNotification(): Promise<Notification | null> {
    console.log('NotificationService: Starting test notification...');
    console.log('NotificationService: Browser support:', this.isSupported);
    console.log('NotificationService: Permission status:', this.permission);
    console.log('NotificationService: Document focus state:', document.hasFocus());
    console.log('NotificationService: Page visibility:', document.visibilityState);

    // Check if the page is focused (some browsers don't show notifications when page is focused)
    const isPageFocused = document.hasFocus();
    const isPageVisible = document.visibilityState === 'visible';

    // Try to show notification even if page is focused (some browsers allow this)
    const notification = await this.showNotification({
      title: '🔔 Test Notification',
      body: `This is a test notification from ARP Guardian. Desktop notifications are working correctly!\nPage focused: ${isPageFocused}\nPage visible: ${isPageVisible}`,
      requireInteraction: false,
      silent: false
    });

    if (notification) {
      console.log('NotificationService: Test notification created successfully');
      
      // Add event listeners to track notification behavior
      notification.onclick = () => {
        console.log('NotificationService: Test notification clicked');
        window.focus();
      };
      
      notification.onshow = () => {
        console.log('NotificationService: Test notification shown');
      };
      
      notification.onclose = () => {
        console.log('NotificationService: Test notification closed');
      };
      
      notification.onerror = (error) => {
        console.error('NotificationService: Test notification error:', error);
      };

      return notification;
    } else {
      console.error('NotificationService: Failed to create test notification');
      return null;
    }
  }

  /**
   * Close all notifications
   */
  closeAll(): void {
    if (this.isSupported) {
      // Note: There's no direct API to close all notifications
      // This would need to be handled by storing notification references
      console.log('Close all notifications requested');
    }
  }
}

// Create a singleton instance
const notificationService = new NotificationService();

export default notificationService; 