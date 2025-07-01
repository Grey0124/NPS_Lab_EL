# ARP Spoofing Detection & Prevention Tool

A comprehensive tool for detecting and preventing ARP spoofing attacks in local networks.

## Features

- Real-time Layer 2 traffic monitoring
- Rule-based and ML-powered anomaly detection
- **Static IP⇄MAC Registry & Rule-Based Checker**
- Automatic prevention measures
- Web-based dashboard for monitoring and control
- Extensible plugin system for additional detection capabilities

## How to Run the Application

### Prerequisites

1. **Run Command Prompt as Administrator**
   - Right-click on Command Prompt and select "Run as administrator"
   - This is required for network packet capture and ARP spoofing tests

2. **Python Version**
   - Ensure you have Python 3.8 or higher installed
   - Check your version: `python --version`

3. **Node.js and npm**
   - Install Node.js from https://nodejs.org/ (version 16 or higher)
   - Verify installation: `node --version` and `npm --version`

### Setup Environment

1. **Clone and Navigate to Project**
   ```bash
   cd /d/NPS_Lab_EL
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python Dependencies**
   ```bash
   # Install main requirements
   pip install -r requirements.txt
   
   # Install backend requirements
   pip install -r arp_app/backend/requirements.txt
   ```

4. **Setup Firebase Configuration**
   - Navigate to the frontend directory: `cd arp_app/frontend/arp_spoof`
   - The `.env` file is already configured with Firebase credentials
   - If you need to update Firebase settings, edit the `.env` file:
   ```env
   VITE_FIREBASE_API_KEY=your_api_key
   VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   VITE_FIREBASE_PROJECT_ID=your_project_id
   VITE_FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
   VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   VITE_FIREBASE_APP_ID=your_app_id
   VITE_FIREBASE_MEASUREMENT_ID=your_measurement_id
   ```

5. **Install Frontend Dependencies**
   ```bash
   cd arp_app/frontend/arp_spoof
   npm install
   ```

### Running the Application

#### 1. Start the Backend Server
```bash
# From the project root directory
cd arp_app/backend
python main.py
```
The backend will start on `http://localhost:8000`

#### 2. Start the Frontend Application
```bash
# From the frontend directory
cd arp_app/frontend/arp_spoof
npm run dev
```
The frontend will start on `http://localhost:5173`

#### 3. Run ARP Spoofing Test
```bash
# From the project root directory
test_arp_spoof.bat
```

### Testing ARP Spoofing Detection

#### Before Running the Test

1. **Check Your Default Gateway**
   ```bash
   # Find your default gateway IP
   ipconfig
   
   # Look for "Default Gateway" in the output
   # Example: Default Gateway . . . . . . . . . : 192.168.1.1
   ```

2. **Verify Network Interface**
   ```bash
   # List available network interfaces
   python src/sniffer.py -l
   ```

3. **Ensure ARP Guardian is Running**
   - Make sure your backend server is running
   - The frontend dashboard should be accessible
   - Monitor the logs for any detection events

#### Running the Test

1. **Execute the Test Script**
   ```bash
   # Run as Administrator
   test_arp_spoof.bat
   ```

2. **What the Test Does**
   - Simulates ARP spoofing attacks
   - Tests the detection capabilities
   - Generates test traffic for monitoring
   - Validates the registry system

3. **Monitor Results**
   - Check the backend logs for detection events
   - View the frontend dashboard for real-time alerts
   - Review log files in the `logs/` directory

### Troubleshooting

#### Common Issues

1. **Permission Denied Errors**
   - Ensure you're running Command Prompt as Administrator
   - Check Windows Defender Firewall settings

2. **Python Package Installation Issues**
   ```bash
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install packages individually if needed
   pip install scapy
   pip install fastapi
   pip install uvicorn
   ```

3. **Frontend Build Issues**
   ```bash
   # Clear npm cache
   npm cache clean --force
   
   # Delete node_modules and reinstall
   rm -rf node_modules
   npm install
   ```

4. **Network Interface Issues**
   - Use `python src/sniffer.py -l` to list available interfaces
   - Try different interface names if the default doesn't work
   - Common Windows interface format: `\Device\NPF_{GUID}`

#### Checking Default Gateway on Windows

```bash
# Method 1: Using ipconfig
ipconfig

# Method 2: Using route command
route print

# Method 3: Using PowerShell
Get-NetRoute -DestinationPrefix "0.0.0.0/0"
```

## Static IP⇄MAC Registry

The tool now includes a static IP⇄MAC registry system that allows you to define known device mappings and detect violations in real-time.

### Registry Configuration

Create a `registry.yml` file with your known device mappings:

```yaml
# Network devices registry
devices:
  "192.168.1.1":
    mac: "00:11:22:33:44:55"
    description: "Main Router"
    device_type: "router"
    trusted: true

  "192.168.1.100":
    mac: "11:22:33:44:55:66"
    description: "Workstation 1"
    device_type: "workstation"
    trusted: true

# Registry settings
settings:
  strict_mode: true  # Only allow registered devices
  log_unknown_devices: true  # Log devices not in registry
  auto_update: false  # Automatically add new devices
  check_interval: 300  # Check registry every 300 seconds
```

### Usage

Run the sniffer with registry checking:

```bash
# Use default registry.yml
python src/sniffer.py -i <interface>

# Specify custom registry file
python src/sniffer.py -i <interface> -r custom_registry.yml

# List available interfaces
python src/sniffer.py -l
```

### Registry Features

- **Compliance Checking**: Each ARP packet is checked against the registry
- **Violation Detection**: Logs warnings when IP→MAC mappings don't match
- **Strict Mode**: Option to only allow registered devices
- **Unknown Device Logging**: Track devices not in the registry
- **Flexible Configuration**: Easy to update and maintain device mappings

## Enhanced CLI Interface

The tool provides a comprehensive command-line interface with advanced logging capabilities.

### Command Line Options

```bash
python src/sniffer.py [OPTIONS]
```

**Available Options:**
- `-i, --interface <interface>`: Network interface to capture packets on (default: eth0)
- `-r, --registry <file>`: Path to registry YAML file (default: registry.yml)
- `--log-file <file>`: Path to log file for mismatches (default: stdout only)
- `-c, --count <number>`: Number of packets to capture (default: capture indefinitely)
- `-l, --list`: List available network interfaces
- `-h, --help`: Show help message

### Usage Examples

**Basic Usage:**
```bash
# Use default settings
python src/sniffer.py -i <interface>

# List available interfaces
python src/sniffer.py -l
```

**With Log File:**
```bash
# Log mismatches to file
python src/sniffer.py -i <interface> --log-file mismatches.log

# Custom registry with log file
python src/sniffer.py -i <interface> -r my_registry.yml --log-file violations.log
```

**Advanced Usage:**
```bash
# Capture specific number of packets
python src/sniffer.py -i <interface> -c 100 --log-file test.log

# Use custom registry and interface
python src/sniffer.py -i "\Device\NPF_{...}" -r network_devices.yml --log-file arp_violations.log
```

### Log File Features

- **Rolling Log Files**: Automatically rotates log files when they reach 10MB
- **Backup Files**: Keeps up to 5 backup files (mismatches.log.1, mismatches.log.2, etc.)
- **Selective Logging**: Only logs warnings and errors to file (INFO messages go to console)
- **UTF-8 Encoding**: Supports international characters in device descriptions
- **Timestamped Entries**: Each log entry includes precise timestamp

### Log File Format

```
2025-06-21 15:03:15 - WARNING - REGISTRY VIOLATION - Source: 192.168.0.1 (98:da:c4:b0:e4:02)
    Warning: MAC mismatch for IP 192.168.0.1
    Device Info: {'mac': 'aa:bb:cc:dd:ee:ff', 'description': 'Router', 'device_type': 'router', 'trusted': True}
2025-06-21 15:03:16 - WARNING - Potential ARP spoofing detected!
    IP: 192.168.0.100
    Old MAC: 11:22:33:44:55:66
    New MAC: aa:bb:cc:dd:ee:ff
```

## Development Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Test registry functionality:
   ```bash
   python test_registry.py
   ```

5. Test CLI interface:
   ```bash
   python test_cli.py
   ```

## Project Structure

- `src/` - Source code
- `tests/` - Test files
- `docs/` - Documentation
- `registry.yml` - Static IP⇄MAC registry configuration
- `test_registry.py` - Registry functionality test script
- `test_cli.py` - CLI interface test script

## License

MIT License 