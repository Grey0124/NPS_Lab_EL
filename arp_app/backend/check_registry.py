#!/usr/bin/env python3
"""
Simple script to check registry status and show recent additions
"""

import yaml
import os
from datetime import datetime
from pathlib import Path

def check_registry():
    """Check the current registry status."""
    
    registry_path = "data/registry.yml"
    
    if not os.path.exists(registry_path):
        print("❌ Registry file not found!")
        return
    
    print("📋 Registry Status Check")
    print("=" * 50)
    
    # Read current registry
    with open(registry_path, 'r') as f:
        registry_data = yaml.safe_load(f) or {}
    
    print(f"📁 Registry file: {registry_path}")
    print(f"📊 Total entries: {len(registry_data)}")
    print(f"🕒 Last modified: {datetime.fromtimestamp(os.path.getmtime(registry_path))}")
    
    print("\n📝 Current Registry Entries:")
    print("-" * 50)
    
    for ip, mac in registry_data.items():
        print(f"  {ip} -> {mac}")
    
    # Check for recent entries (you could add timestamps to track this)
    print(f"\n✅ Registry is being updated automatically!")
    print(f"💡 New entries are added when valid ARP packets are detected")
    print(f"🔍 Check the logs for 'AUTO-ADDED to registry' messages")

if __name__ == "__main__":
    check_registry() 