#!/usr/bin/env python3
"""
Test script for the ARP Sniffer with Registry Checking functionality.
This script demonstrates the registry loading and compliance checking features.
"""

import sys
import os
import yaml
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sniffer import ARPSniffer

def test_registry_loading():
    """Test registry loading functionality."""
    print("=== Testing Registry Loading ===")
    
    # Test with existing registry file
    sniffer = ARPSniffer(interface="eth0", registry_file="registry.yml")
    
    print(f"Loaded {len(sniffer.registry_devices)} devices from registry")
    print(f"Registry settings: {sniffer.registry_settings}")
    
    # Display loaded devices
    print("\nRegistered devices:")
    for ip, device in sniffer.registry_devices.items():
        print(f"  {ip} -> {device['mac']} ({device['description']})")

def test_compliance_checking():
    """Test compliance checking functionality."""
    print("\n=== Testing Compliance Checking ===")
    
    sniffer = ARPSniffer(interface="eth0", registry_file="registry.yml")
    
    # Test cases
    test_cases = [
        # (ip, mac, expected_result)
        ("192.168.1.1", "00:11:22:33:44:55", "compliant"),  # Correct mapping
        ("192.168.1.1", "aa:bb:cc:dd:ee:ff", "violation"),  # Wrong MAC
        ("192.168.1.100", "11:22:33:44:55:66", "compliant"),  # Correct mapping
        ("192.168.1.999", "99:88:77:66:55:44", "unknown"),  # Unknown IP
    ]
    
    for ip, mac, expected in test_cases:
        result = sniffer._check_registry_compliance(ip, mac)
        status = "✓" if result['compliant'] else "✗"
        print(f"{status} {ip} -> {mac}")
        print(f"    Result: {result['warning'] or 'Compliant'}")
        print()

def create_test_registry():
    """Create a test registry file for demonstration."""
    test_registry = {
        'devices': {
            '192.168.1.1': {
                'mac': '00:11:22:33:44:55',
                'description': 'Test Router',
                'device_type': 'router',
                'trusted': True
            },
            '192.168.1.100': {
                'mac': '11:22:33:44:55:66',
                'description': 'Test Workstation',
                'device_type': 'workstation',
                'trusted': True
            }
        },
        'settings': {
            'strict_mode': True,
            'log_unknown_devices': True,
            'auto_update': False,
            'check_interval': 300
        }
    }
    
    with open('test_registry.yml', 'w') as f:
        yaml.dump(test_registry, f, default_flow_style=False)
    
    print("Created test_registry.yml")

def main():
    """Main test function."""
    print("ARP Sniffer Registry Testing")
    print("=" * 40)
    
    # Create test registry if it doesn't exist
    if not os.path.exists('registry.yml'):
        print("Registry file not found. Creating test registry...")
        create_test_registry()
    
    # Run tests
    test_registry_loading()
    test_compliance_checking()
    
    print("\n=== Usage Example ===")
    print("To run the sniffer with registry checking:")
    print("  python src/sniffer.py -i <interface> -r registry.yml")
    print("\nTo list available interfaces:")
    print("  python src/sniffer.py -l")

if __name__ == "__main__":
    main() 