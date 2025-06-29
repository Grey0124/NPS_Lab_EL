# ARP Guardian - ARP Spoofing Detection System

A comprehensive ARP spoofing detection system with a modern React frontend and FastAPI backend, featuring real-time monitoring, machine learning detection, and an intuitive web interface.

## Features

- **Real-time ARP Monitoring**: Continuously monitor network interfaces for ARP spoofing attacks
- **Machine Learning Detection**: Advanced ML-powered threat detection with configurable sensitivity
- **Rule-based Detection**: Traditional rule-based detection for known attack patterns
- **WebSocket Real-time Updates**: Live updates via WebSocket for instant threat notifications
- **Modern Web Interface**: Beautiful, responsive React frontend with blue/purple theme
- **Comprehensive Statistics**: Detailed analytics and performance metrics
- **Alert Management**: Complete alert lifecycle management with acknowledgment and resolution
- **Configuration Management**: Flexible system configuration with real-time updates

## Architecture

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **Real-time Communication**: WebSocket
- **Detection Engine**: Custom ARP spoofing detector with ML integration

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Network interface access (for monitoring)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd arp_app/backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd arp_app/frontend/arp_spoof
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

### 1. Home Page
- Overview of the system capabilities
- Quick access to monitoring and configuration
- System statistics and feature highlights

### 2. Monitoring Page
- Select network interface for monitoring
- Start/stop real-time ARP monitoring
- View live detection feed with threat details
- Real-time statistics and status updates

### 3. Statistics Page
- Comprehensive detection analytics
- Performance metrics and trends
- Recent detection history
- Configurable time ranges

### 4. Configuration Page
- Detection sensitivity settings
- Alert configuration (email, webhook)
- System preferences
- Real-time configuration updates

### 5. Alerts Page
- View and manage security alerts
- Acknowledge and resolve threats
- Alert filtering and statistics
- Alert history and trends

## API Endpoints

### Monitoring
- `POST /api/v1/monitoring/start` - Start monitoring on interface
- `POST /api/v1/monitoring/stop` - Stop monitoring
- `GET /api/v1/monitoring/status` - Get monitoring status
- `GET /api/v1/monitoring/interfaces` - List available interfaces

### Statistics
- `GET /api/v1/statistics` - Get detection statistics
- `GET /api/v1/detections` - Get recent detections

### Configuration
- `GET /api/v1/config` - Get current configuration
- `PUT /api/v1/config` - Update configuration
- `POST /api/v1/config/reset` - Reset to defaults

### Alerts
- `GET /api/v1/alerts` - Get alert history
- `GET /api/v1/alerts/stats` - Get alert statistics
- `DELETE /api/v1/alerts` - Clear alert history

### WebSocket
- `WS /ws` - Real-time updates for monitoring status, detections, and alerts

## Configuration

### Detection Settings
- **Sensitivity**: Low, Medium, High, Critical
- **Scan Interval**: Packet analysis frequency
- **ML Detection**: Enable/disable machine learning
- **Heuristics**: Enable/disable rule-based detection

### Alert Settings
- **Email Alerts**: Configure email notifications
- **Webhook Alerts**: Configure webhook endpoints
- **Desktop Notifications**: Browser notifications
- **Sound Alerts**: Audio notifications

### System Settings
- **Log Level**: Debug, Info, Warning, Error
- **Auto-start**: Start monitoring on system boot
- **Backup**: Automatic configuration backup

## Development

### Backend Development
- FastAPI with automatic API documentation at `/docs`
- Modular service architecture
- Comprehensive error handling
- WebSocket integration for real-time updates

### Frontend Development
- React with TypeScript for type safety
- Tailwind CSS for styling
- React Router for navigation
- Custom API service for backend communication

## Security Features

- Real-time ARP table monitoring
- Machine learning-based threat detection
- Rule-based attack pattern recognition
- Configurable detection thresholds
- Comprehensive alerting system
- Audit logging and statistics

## Troubleshooting

### Backend Issues
- Ensure Python dependencies are installed
- Check network interface permissions
- Verify FastAPI server is running on port 8000
- Check logs for detailed error messages

### Frontend Issues
- Ensure Node.js dependencies are installed
- Check API endpoint connectivity
- Verify WebSocket connection
- Check browser console for errors

### Monitoring Issues
- Ensure network interface exists and is accessible
- Check for sufficient permissions to capture packets
- Verify interface is active and has traffic
- Check detection configuration settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Check the logs for detailed error messages
- Ensure all dependencies are properly installed 