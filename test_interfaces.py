#!/usr/bin/env python3
"""
Test script to check network interfaces using different methods
"""

import platform
import subprocess
import sys

def test_scapy_interfaces():
    """Test Scapy interface detection."""
    try:
        from scapy.all import get_if_list, get_if_addr, get_if_hwaddr
        print("=== Scapy Interfaces ===")
        interfaces = get_if_list()
        for iface in interfaces:
            try:
                ip = get_if_addr(iface)
                mac = get_if_hwaddr(iface)
                print(f"  {iface} - IP: {ip}, MAC: {mac}")
            except Exception as e:
                print(f"  {iface} - Error getting IP/MAC: {e}")
        return interfaces
    except ImportError:
        print("Scapy not available")
        return []

def test_psutil_interfaces():
    """Test psutil interface detection."""
    try:
        import psutil
        print("\n=== psutil Interfaces ===")
        interfaces = psutil.net_if_addrs()
        for name, addrs in interfaces.items():
            print(f"  {name}:")
            for addr in addrs:
                print(f"    {addr.family.name}: {addr.address}")
        return list(interfaces.keys())
    except ImportError:
        print("psutil not available")
        return []

def test_netifaces_interfaces():
    """Test netifaces interface detection."""
    try:
        import netifaces
        print("\n=== netifaces Interfaces ===")
        interfaces = netifaces.interfaces()
        for iface in interfaces:
            try:
                addrs = netifaces.ifaddresses(iface)
                print(f"  {iface}: {addrs}")
            except Exception as e:
                print(f"  {iface}: Error - {e}")
        return interfaces
    except ImportError:
        print("netifaces not available")
        return []

def test_windows_netsh():
    """Test Windows netsh command for interface names."""
    if platform.system() == "Windows":
        print("\n=== Windows netsh Interfaces ===")
        try:
            result = subprocess.run(['netsh', 'interface', 'show', 'interface'], 
                                  capture_output=True, text=True, shell=True)
            print(result.stdout)
        except Exception as e:
            print(f"Error running netsh: {e}")

def test_ipconfig():
    """Test ipconfig command."""
    if platform.system() == "Windows":
        print("\n=== Windows ipconfig ===")
        try:
            result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
            print(result.stdout)
        except Exception as e:
            print(f"Error running ipconfig: {e}")

if __name__ == "__main__":
    print(f"Platform: {platform.system()}")
    print(f"Python version: {sys.version}")
    
    test_scapy_interfaces()
    test_psutil_interfaces()
    test_netifaces_interfaces()
    test_windows_netsh()
    test_ipconfig() 