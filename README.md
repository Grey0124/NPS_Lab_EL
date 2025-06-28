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