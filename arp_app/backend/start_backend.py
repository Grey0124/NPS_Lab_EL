#!/usr/bin/env python3
"""
Startup script for ARP Detection Backend
Tests basic functionality before starting the main server
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Package name to import name mapping
    package_imports = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'scapy': 'scapy',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'joblib': 'joblib',
        'pyyaml': 'yaml'  # PyYAML imports as yaml
    }
    
    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    required_dirs = [
        'data', 'logs', 'models'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"📁 Creating {dir_name}/")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

def test_backend_startup():
    """Test if the backend can start without errors."""
    print("\nTesting backend startup...")
    
    try:
        # Try to import main modules
        import main
        print("✅ Main module imports successfully")
        
        # Try to create FastAPI app
        from main import app
        print("✅ FastAPI app created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Backend startup test failed: {e}")
        return False

def main():
    """Main startup function."""
    print("🚀 ARP Detection Backend Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check directories
    if not check_directories():
        sys.exit(1)
    
    # Test backend startup
    if not test_backend_startup():
        sys.exit(1)
    
    print("\n✅ All checks passed! Starting backend server...")
    print("=" * 50)
    
    # Start the main server
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 