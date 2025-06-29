#!/usr/bin/env python3
"""
Test script to demonstrate automatic registry addition functionality
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from services.arp_detector import ARPSpoofingDetector
from services.arp_registry import ARPRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_auto_registry_addition():
    """Test automatic registry addition functionality."""
    
    print("🧪 Testing Automatic Registry Addition")
    print("=" * 50)
    
    # Create a test registry
    test_registry_path = "data/test_registry.yml"
    registry = ARPRegistry(test_registry_path)
    
    # Clear any existing entries
    registry.reset()
    print(f"✅ Created test registry at: {test_registry_path}")
    
    # Create detector with auto registry addition enabled
    detector = ARPSpoofingDetector(
        interface=None,  # We won't actually capture packets
        registry_path=test_registry_path,
        auto_registry_addition=True
    )
    
    print(f"✅ Created detector with auto_registry_addition: {detector.auto_registry_addition}")
    
    # Test valid ARP entries (private IPs)
    test_entries = [
        ("192.168.1.100", "00:11:22:33:44:55"),
        ("192.168.1.101", "00:11:22:33:44:66"),
        ("10.0.0.50", "aa:bb:cc:dd:ee:ff"),
        ("172.16.1.10", "11:22:33:44:55:66")
    ]
    
    print("\n📝 Testing automatic addition of valid entries:")
    for ip, mac in test_entries:
        is_valid = detector.is_valid_arp_entry(ip, mac)
        print(f"  {ip} -> {mac}: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if is_valid:
            added = detector.registry_manager.add_entry(ip, mac)
            if added:
                print(f"    ✅ Added to registry")
            else:
                print(f"    ℹ️  Already exists in registry")
    
    # Test invalid entries (public IPs)
    invalid_entries = [
        ("8.8.8.8", "00:11:22:33:44:77"),  # Public IP
        ("1.1.1.1", "aa:bb:cc:dd:ee:88"),  # Public IP
    ]
    
    print("\n🚫 Testing rejection of invalid entries:")
    for ip, mac in invalid_entries:
        is_valid = detector.is_valid_arp_entry(ip, mac)
        print(f"  {ip} -> {mac}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Show final registry state
    print("\n📊 Final Registry State:")
    entries = registry.list_entries()
    for ip, mac in entries.items():
        print(f"  {ip} -> {mac}")
    
    print(f"\n📈 Registry Statistics:")
    stats = detector.get_registry_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up
    registry.reset()
    print(f"\n🧹 Cleaned up test registry")
    
    print("\n✅ Automatic registry addition test completed!")

if __name__ == "__main__":
    asyncio.run(test_auto_registry_addition()) 