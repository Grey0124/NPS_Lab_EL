#!/usr/bin/env python3
"""
CLI tool for managing the ARP registry (data/registry.yml)
"""
import sys
from arp_app.backend.services.arp_registry import ARPRegistry

def print_usage():
    print("""
Usage:
  python manage_registry.py list
  python manage_registry.py add <ip> <mac>
  python manage_registry.py remove <ip>
  python manage_registry.py reset
""")

def main():
    registry = ARPRegistry()
    if len(sys.argv) < 2:
        print_usage()
        return
    cmd = sys.argv[1]
    if cmd == 'list':
        entries = registry.list_entries()
        if not entries:
            print("Registry is empty.")
        else:
            print("Current ARP Registry:")
            for ip, mac in entries.items():
                print(f"  {ip} -> {mac}")
    elif cmd == 'add' and len(sys.argv) == 4:
        ip, mac = sys.argv[2], sys.argv[3]
        if registry.add_entry(ip, mac):
            print(f"Added {ip} -> {mac} to registry.")
        else:
            print(f"Entry {ip} -> {mac} already exists.")
    elif cmd == 'remove' and len(sys.argv) == 3:
        ip = sys.argv[2]
        if registry.remove_entry(ip):
            print(f"Removed {ip} from registry.")
        else:
            print(f"No entry for {ip} found.")
    elif cmd == 'reset':
        registry.reset()
        print("Registry reset (all entries removed).")
    else:
        print_usage()

if __name__ == '__main__':
    main() 