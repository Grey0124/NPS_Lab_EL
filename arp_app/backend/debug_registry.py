#!/usr/bin/env python3
"""
Debug script to test registry service directly
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.registry_service import RegistryService
from services.arp_registry import ARPRegistry

async def debug_registry():
    """Debug the registry service."""
    print("🔍 Debugging Registry Service...")
    
    try:
        # Test the registry service
        service = RegistryService()
        entries = await service.get_entries()
        
        print(f"📊 Registry Service entries: {len(entries)}")
        print("📝 Registry entries from service:")
        print("-" * 50)
        
        for ip, mac in entries.items():
            print(f"  {ip}: {mac}")
        
        print("-" * 50)
        
        # Test the ARPRegistry directly
        print("\n🔍 Testing ARPRegistry directly...")
        registry = ARPRegistry()
        direct_entries = registry.list_entries()
        
        print(f"📊 Direct ARPRegistry entries: {len(direct_entries)}")
        print("📝 Direct registry entries:")
        print("-" * 50)
        
        for ip, mac in direct_entries.items():
            print(f"  {ip}: {mac}")
        
        print("-" * 50)
        
        # Check if they match
        if len(entries) == len(direct_entries):
            print("✅ Service and direct registry match!")
        else:
            print(f"❌ Mismatch! Service: {len(entries)}, Direct: {len(direct_entries)}")
            
        # Check for missing entries
        service_ips = set(entries.keys())
        direct_ips = set(direct_entries.keys())
        
        missing_in_service = direct_ips - service_ips
        if missing_in_service:
            print(f"❌ Missing in service: {missing_in_service}")
        
        extra_in_service = service_ips - direct_ips
        if extra_in_service:
            print(f"❌ Extra in service: {extra_in_service}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_registry()) 