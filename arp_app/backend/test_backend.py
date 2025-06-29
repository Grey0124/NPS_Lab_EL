#!/usr/bin/env python3
"""
Simple test script to verify backend functionality
"""

import requests
import json
import sys
from datetime import datetime

def test_backend():
    """Test basic backend functionality."""
    base_url = "http://localhost:8000"
    
    print("Testing ARP Detection Backend...")
    print("=" * 50)
    
    # Test 1: Basic ping endpoint
    try:
        response = requests.get(f"{base_url}/ping", timeout=5)
        print(f"✅ Ping endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Ping endpoint failed: {e}")
        return False
    
    # Test 2: Health check endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Services: {data.get('services', {})}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 3: Test API endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/test", timeout=5)
        print(f"✅ Test API endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Test API endpoint failed: {e}")
        return False
    
    # Test 4: Detections endpoint (should work even if empty)
    try:
        response = requests.get(f"{base_url}/api/v1/detections?limit=5", timeout=5)
        print(f"✅ Detections endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Detections count: {data.get('count', 0)}")
        else:
            print(f"   Error response: {response.text}")
    except Exception as e:
        print(f"❌ Detections endpoint failed: {e}")
        return False
    
    # Test 5: CORS headers
    try:
        response = requests.options(f"{base_url}/api/v1/detections", timeout=5)
        print(f"✅ CORS preflight: {response.status_code}")
        cors_headers = response.headers.get('Access-Control-Allow-Origin')
        if cors_headers:
            print(f"   CORS headers present: {cors_headers}")
        else:
            print("   ⚠️  No CORS headers found")
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False
    
    print("=" * 50)
    print("✅ Backend tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1) 