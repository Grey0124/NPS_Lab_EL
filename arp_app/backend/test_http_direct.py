#!/usr/bin/env python3
"""
Direct HTTP test to check what the API endpoint returns
"""

import requests
import json

def test_http_direct():
    """Test the HTTP API endpoint directly."""
    print("🔍 Testing HTTP API endpoint directly...")
    
    try:
        # Test the registry endpoint
        response = requests.get('http://localhost:8000/api/v1/registry', timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            entries = data.get('entries', {})
            count = data.get('count', 0)
            
            print(f"📊 API Response count: {count}")
            print(f"📊 Actual entries count: {len(entries)}")
            print("📝 All entries from API:")
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
            
            print(f"\n📊 Summary: API returned {len(entries)} entries, expected 14")
            
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
    test_http_direct() 