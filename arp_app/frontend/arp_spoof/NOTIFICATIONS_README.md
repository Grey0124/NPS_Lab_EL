# Desktop Notifications for ARP Guardian

## Overview

The Alerts page now includes comprehensive desktop notification functionality to keep users informed about security threats in real-time.

## Features

### 🔔 Desktop Notifications
- **Browser Permission Management**: Automatic permission requests and status tracking
- **Test Notifications**: Built-in test functionality to verify notification setup
- **Configurable Settings**: Granular control over notification preferences

### ⚙️ Notification Settings

#### Alert Severity Filtering
- **Critical**: Always enabled by default (highest priority)
- **High**: Enabled by default for important threats
- **Medium**: Enabled by default for moderate threats
- **Low**: Disabled by default (can be enabled if needed)

#### Notification Behavior
- **Require Interaction**: Critical and high-severity alerts require user acknowledgment
- **Sound**: Play notification sounds for audio alerts
- **Auto-close**: Non-critical notifications auto-close after 10 seconds

### 🔄 Real-time Updates
- **WebSocket Integration**: Real-time alert delivery via WebSocket connection
- **Live Status Monitoring**: Connection status indicators
- **Automatic Reconnection**: Robust reconnection logic with exponential backoff

## Usage

### Enabling Notifications
1. Navigate to the **Alerts** page
2. Click **"Enable Notifications"** if not already enabled
3. Grant browser permission when prompted
4. Use **"Test Notification"** to verify setup

### Configuring Preferences
1. Click **"Show Settings"** in the notification panel
2. Adjust severity filters as needed
3. Configure notification behavior preferences
4. Settings are applied immediately

### Real-time Monitoring
- **Connection Status**: Green dot indicates active WebSocket connection
- **Enable/Disable**: Toggle real-time updates as needed
- **Automatic Updates**: New alerts appear instantly without page refresh

## Technical Implementation

### Services
- **`notificationService.ts`**: Handles browser notification API
- **`websocket.ts`**: Manages real-time WebSocket connections
- **Integration**: Seamless integration with existing alert management

### Browser Compatibility
- **Supported**: Chrome, Firefox, Safari, Edge (modern versions)
- **Fallback**: Graceful degradation for unsupported browsers
- **Permission Handling**: Proper permission state management

### Security Features
- **Permission-based**: Only shows notifications with explicit user consent
- **Configurable**: Users control which alerts trigger notifications
- **Non-intrusive**: Respects user preferences and browser settings

## Troubleshooting

### Notifications Not Working
1. Check browser notification permissions
2. Verify notification settings are enabled
3. Test with the "Test Notification" button
4. Check browser console for errors

### WebSocket Connection Issues
1. Verify backend WebSocket server is running
2. Check network connectivity
3. Review browser console for connection errors
4. Try disabling/enabling real-time updates

### Permission Denied
1. Click browser notification icon in address bar
2. Select "Allow" for notifications
3. Refresh the page and try again
4. Check browser settings for site permissions

## Best Practices

### For Users
- Enable notifications for critical and high-severity alerts
- Keep real-time updates enabled for immediate threat awareness
- Test notifications after initial setup
- Review notification settings periodically

### For Developers
- Monitor WebSocket connection status
- Handle notification permission changes gracefully
- Provide clear feedback for notification states
- Implement proper cleanup on component unmount

## Future Enhancements

- **Notification Actions**: Click-to-acknowledge from notification
- **Custom Sounds**: Different sounds for different alert severities
- **Notification History**: Track notification delivery and interaction
- **Mobile Support**: Push notifications for mobile devices
- **Advanced Filtering**: Time-based and custom notification rules 