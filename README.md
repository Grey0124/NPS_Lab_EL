# ARP Spoofing Detection & Prevention Tool

A comprehensive tool for detecting and preventing ARP spoofing attacks in local networks.

## Features

- Real-time Layer 2 traffic monitoring
- Rule-based and ML-powered anomaly detection
- **Static IP⇄MAC Registry & Rule-Based Checker**
- Automatic prevention measures
- Web-based dashboard for monitoring and control
- Extensible plugin system for additional detection capabilities

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

## Project Structure

- `src/` - Source code
- `tests/` - Test files
- `docs/` - Documentation
- `registry.yml` - Static IP⇄MAC registry configuration
- `test_registry.py` - Registry functionality test script

## License

MIT License 