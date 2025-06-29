#!/usr/bin/env python3
"""
Test script to check registry API endpoint
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.registry_service import RegistryService

async def test_registry():
    """Test the registry service."""
    print("🔍 Testing Registry Service...")
    
    try:
        # Create registry service
        service = RegistryService()
        
        # Get entries
        entries = await service.get_entries()
        
        print(f"📊 Total entries: {len(entries)}")
        print("📝 Registry entries:")
        print("-" * 50)
        
        for ip, mac in entries.items():
            print(f"  {ip}: {mac}")
        
        print("-" * 50)
        print(f"✅ Registry service working correctly")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_registry()) 