#!/usr/bin/env python3
"""
Test script to verify ARP table parsing
"""

import subprocess
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_arp_table_parsing():
    """Test ARP table parsing."""
    try:
        print("Testing ARP table parsing...")
        print("=" * 50)
        
        # Get raw ARP table output
        print("Raw ARP table output:")
        print("-" * 30)
        result = subprocess.run(['arp', '-a'], capture_output=True, text=True, check=True)
        print(result.stdout)
        print("=" * 50)
        
        # Test parsing logic
        print("Parsed ARP table entries:")
        print("-" * 30)
        
        arp_entries = []
        seen_entries = set()
        current_interface = None
        
        lines = result.stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            print(f"Processing line: '{line}'")
            
            # Check if this is an interface header line
            if line.startswith('Interface:'):
                # Extract interface information
                # Format: "Interface: 192.168.59.1 --- 0xa"
                interface_parts = line.split('---')
                if len(interface_parts) >= 1:
                    interface_info = interface_parts[0].replace('Interface:', '').strip()
                    # Extract IP address from interface info
                    interface_ip = interface_info.split()[0] if interface_info else None
                    current_interface = interface_ip
                    print(f"  -> Found interface: {current_interface}")
                continue
            
            # Check if this is a data line with IP and MAC
            if 'dynamic' in line.lower() or 'static' in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[0]
                    mac = parts[1]
                    entry_type = 'dynamic' if 'dynamic' in line.lower() else 'static'
                    
                    # Create unique identifier to avoid duplicates
                    entry_key = f"{ip}-{mac}"
                    if entry_key not in seen_entries:
                        seen_entries.add(entry_key)
                        entry = {
                            'ip': ip,
                            'mac': mac,
                            'type': entry_type,
                            'interface': current_interface,
                            'is_legitimate': False  # We don't have legitimate entries in this test
                        }
                        arp_entries.append(entry)
                        print(f"  -> Added entry: {ip} -> {mac} (type: {entry_type}, interface: {current_interface})")
        
        print("=" * 50)
        print(f"Total entries found: {len(arp_entries)}")
        print("Final ARP table:")
        print("-" * 30)
        
        for i, entry in enumerate(arp_entries, 1):
            print(f"{i:2d}. {entry['ip']:15s} -> {entry['mac']:17s} ({entry['type']:7s}) [{entry['interface']}]")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_arp_table_parsing() 