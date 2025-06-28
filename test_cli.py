#!/usr/bin/env python3
"""
Test script for the enhanced CLI interface with log file functionality.
This script demonstrates the new argparse options and rolling logfile features.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def test_cli_help():
    """Test the help output of the CLI."""
    print("=== Testing CLI Help ===")
    try:
        result = subprocess.run([
            sys.executable, "src/sniffer.py", "--help"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        print("Help output:")
        print(result.stdout)
        
        if result.returncode == 0:
            print("✓ Help command works correctly")
        else:
            print("✗ Help command failed")
            
    except Exception as e:
        print(f"✗ Error running help command: {e}")

def test_list_interfaces():
    """Test the list interfaces functionality."""
    print("\n=== Testing List Interfaces ===")
    try:
        result = subprocess.run([
            sys.executable, "src/sniffer.py", "-l"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✓ List interfaces command works")
            print("Available interfaces found:")
            # Show first few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... and {len(lines) - 10} more lines")
        else:
            print("✗ List interfaces command failed")
            print("Error output:", result.stderr)
            
    except Exception as e:
        print(f"✗ Error running list interfaces command: {e}")

def test_log_file_creation():
    """Test log file creation and rolling functionality."""
    print("\n=== Testing Log File Creation ===")
    
    # Create a test log file
    test_log_file = "test_mismatches.log"
    
    try:
        # Run sniffer for a short time with log file
        print(f"Creating log file: {test_log_file}")
        
        # Start the sniffer in background for a few seconds
        process = subprocess.Popen([
            sys.executable, "src/sniffer.py", 
            "-i", "eth0",  # This will likely fail, but we're testing CLI parsing
            "--log-file", test_log_file,
            "-c", "1"  # Capture only 1 packet
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        
        # Wait a moment for the process to start
        time.sleep(2)
        
        # Check if log file was created
        if os.path.exists(test_log_file):
            print(f"✓ Log file created: {test_log_file}")
            
            # Read the log file
            with open(test_log_file, 'r') as f:
                content = f.read()
                print(f"Log file content ({len(content)} characters):")
                if content:
                    print("  " + content.replace('\n', '\n  '))
                else:
                    print("  (empty)")
        else:
            print(f"✗ Log file not created: {test_log_file}")
        
        # Clean up
        process.terminate()
        if os.path.exists(test_log_file):
            os.remove(test_log_file)
            print(f"Cleaned up: {test_log_file}")
            
    except Exception as e:
        print(f"✗ Error testing log file creation: {e}")

def test_registry_file_option():
    """Test the registry file option."""
    print("\n=== Testing Registry File Option ===")
    
    # Create a test registry file
    test_registry = "test_registry_cli.yml"
    test_registry_content = """
devices:
  "192.168.1.1":
    mac: "00:11:22:33:44:55"
    description: "Test Router"
    device_type: "router"
    trusted: true
settings:
  strict_mode: false
  log_unknown_devices: true
  auto_update: false
  check_interval: 300
"""
    
    try:
        # Create test registry file
        with open(test_registry, 'w') as f:
            f.write(test_registry_content)
        print(f"✓ Created test registry: {test_registry}")
        
        # Test CLI with custom registry
        result = subprocess.run([
            sys.executable, "src/sniffer.py", 
            "-r", test_registry,
            "-l"  # Just list interfaces to test registry loading
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✓ Custom registry file option works")
        else:
            print("✗ Custom registry file option failed")
            print("Error output:", result.stderr)
            
        # Clean up
        if os.path.exists(test_registry):
            os.remove(test_registry)
            print(f"Cleaned up: {test_registry}")
            
    except Exception as e:
        print(f"✗ Error testing registry file option: {e}")

def main():
    """Main test function."""
    print("Enhanced CLI Interface Testing")
    print("=" * 50)
    
    # Run tests
    test_cli_help()
    test_list_interfaces()
    test_log_file_creation()
    test_registry_file_option()
    
    print("\n=== Usage Examples ===")
    print("Basic usage:")
    print("  python src/sniffer.py -i <interface>")
    print("\nWith log file:")
    print("  python src/sniffer.py -i <interface> --log-file mismatches.log")
    print("\nWith custom registry:")
    print("  python src/sniffer.py -i <interface> -r my_registry.yml --log-file violations.log")
    print("\nList interfaces:")
    print("  python src/sniffer.py -l")
    print("\nCapture specific number of packets:")
    print("  python src/sniffer.py -i <interface> -c 100 --log-file test.log")

if __name__ == "__main__":
    main() 