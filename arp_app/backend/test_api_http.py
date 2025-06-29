#!/usr/bin/env python3
"""
Test script to check the HTTP API endpoint for registry entries
"""

import requests
import json

def test_registry_api():
    """Test the registry API endpoint via HTTP."""
    print("🔍 Testing Registry API Endpoint...")
    
    try:
        # Test the registry endpoint with correct prefix
        response = requests.get('http://localhost:8000/api/v1/registry', timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response Data: {json.dumps(data, indent=2)}")
            
            entries = data.get('entries', {})
            count = data.get('count', 0)
            
            print(f"📊 Total entries from API: {count}")
            print(f"📊 Actual entries count: {len(entries)}")
            print("📝 Registry entries from API:")
            print("-" * 50)
            
            for ip, mac in entries.items():
                print(f"  {ip}: {mac}")
            
            print("-" * 50)
            print(f"✅ API endpoint working correctly")
            
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
    test_registry_api() 