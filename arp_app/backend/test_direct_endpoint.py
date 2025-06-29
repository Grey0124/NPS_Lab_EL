#!/usr/bin/env python3
"""
Test script for the new direct registry endpoint
"""

import requests
import json

def test_direct_endpoint():
    """Test the new direct registry endpoint."""
    print("🔍 Testing Direct Registry Endpoint...")
    
    try:
        # Test the new direct registry endpoint
        response = requests.get('http://localhost:8000/api/v1/registry/direct', timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            entries = data.get('entries', {})
            count = data.get('count', 0)
            source = data.get('source', 'unknown')
            
            print(f"📊 API Response count: {count}")
            print(f"📊 Actual entries count: {len(entries)}")
            print(f"📊 Source: {source}")
            print("📝 All entries from direct endpoint:")
            print("-" * 50)
            
            for i, (ip, mac) in enumerate(entries.items(), 1):
                print(f"  {i:2d}. {ip}: {mac}")
            
            print("-" * 50)
            
            # Check for specific missing entries
            expected_entries = [
                "192.168.188.241",
                "192.168.188.253"
            ]
            
            print("\n🔍 Checking for missing entries:")
            for expected_ip in expected_entries:
                if expected_ip in entries:
                    print(f"  ✅ {expected_ip}: {entries[expected_ip]}")
                else:
                    print(f"  ❌ {expected_ip}: MISSING")
            
            print(f"\n📊 Summary: Direct endpoint returned {len(entries)} entries, expected 14")
            
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"❌ Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend server")
        print("💡 Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_endpoint() 