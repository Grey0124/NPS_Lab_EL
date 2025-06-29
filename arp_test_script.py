#!/usr/bin/env python3
"""
ARP Spoofing Test Script for Windows
This script will send fake ARP responses to test the detection system.
Run this script while your ARP Guardian is monitoring the network.
"""

import time
import socket
import struct
import binascii
from scapy.all import *

def create_arp_packet(src_ip, src_mac, dst_ip, dst_mac):
    """Create an ARP packet."""
    # Create Ethernet frame
    ethernet = Ether(dst=dst_mac, src=src_mac, type=0x0806)
    
    # Create ARP packet
    arp = ARP(
        hwtype=1,  # Ethernet
        ptype=0x0800,  # IP
        hwlen=6,  # MAC address length
        plen=4,  # IP address length
        op=2,  # ARP reply
        hwsrc=src_mac,
        psrc=src_ip,
        hwdst=dst_mac,
        pdst=dst_ip
    )
    
    return ethernet / arp

def get_interface_info():
    """Get network interface information."""
    interfaces = get_if_list()
    print("Available interfaces:")
    for i, iface in enumerate(interfaces):
        try:
            ip = get_if_addr(iface)
            mac = get_if_hwaddr(iface)
            print(f"{i}: {iface} - IP: {ip}, MAC: {mac}")
        except:
            print(f"{i}: {iface} - No IP/MAC info")
    
    return interfaces

def test_arp_spoofing():
    """Test ARP spoofing detection."""
    print("=== ARP Spoofing Test Script ===")
    print("This script will send fake ARP responses to test your detection system.")
    print("Make sure your ARP Guardian is running and monitoring the network.\n")
    
    # Get interface information
    interfaces = get_interface_info()
    
    # Let user select interface
    try:
        choice = int(input("\nSelect interface number: "))
        if choice < 0 or choice >= len(interfaces):
            print("Invalid choice!")
            return
        interface = interfaces[choice]
    except ValueError:
        print("Invalid input!")
        return
    
    # Get interface details
    try:
        src_ip = get_if_addr(interface)
        src_mac = get_if_hwaddr(interface)
        print(f"\nUsing interface: {interface}")
        print(f"Source IP: {src_ip}")
        print(f"Source MAC: {src_mac}")
    except Exception as e:
        print(f"Error getting interface info: {e}")
        return
    
    # Get target information
    try:
        target_ip = input("Enter target IP (e.g., 192.168.1.1 for router): ").strip()
        if not target_ip:
            print("No target IP provided!")
            return
        
        # Get target MAC (optional)
        try:
            target_mac = getmacbyip(target_ip)
            print(f"Target MAC: {target_mac}")
        except:
            target_mac = "ff:ff:ff:ff:ff:ff"  # Broadcast
            print(f"Could not resolve target MAC, using broadcast: {target_mac}")
        
        # Get fake MAC for spoofing
        fake_mac = input("Enter fake MAC to spoof (e.g., 00:11:22:33:44:55): ").strip()
        if not fake_mac:
            fake_mac = "00:11:22:33:44:55"  # Default fake MAC
            print(f"Using default fake MAC: {fake_mac}")
        
    except Exception as e:
        print(f"Error getting target info: {e}")
        return
    
    # Confirm before starting
    print(f"\n=== Test Configuration ===")
    print(f"Interface: {interface}")
    print(f"Source IP: {src_ip}")
    print(f"Source MAC: {src_mac}")
    print(f"Target IP: {target_ip}")
    print(f"Target MAC: {target_mac}")
    print(f"Fake MAC: {fake_mac}")
    
    confirm = input("\nStart ARP spoofing test? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return
    
    print(f"\nStarting ARP spoofing test...")
    print("Press Ctrl+C to stop the test.")
    
    try:
        packet_count = 0
        while True:
            # Create fake ARP response
            arp_packet = create_arp_packet(
                src_ip=target_ip,  # Spoofing target IP
                src_mac=fake_mac,  # Using fake MAC
                dst_ip=src_ip,     # Sending to our IP
                dst_mac=src_mac    # Our MAC
            )
            
            # Send the packet
            sendp(arp_packet, iface=interface, verbose=False)
            packet_count += 1
            
            print(f"Sent ARP spoof packet #{packet_count} - {target_ip} -> {fake_mac}")
            
            # Wait a bit between packets
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\nTest stopped. Sent {packet_count} ARP spoof packets.")
        print("Check your ARP Guardian dashboard for detections!")

if __name__ == "__main__":
    # Check if running as administrator
    try:
        # Try to create a raw socket (requires admin privileges)
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        test_socket.close()
    except PermissionError:
        print("ERROR: This script requires Administrator privileges!")
        print("Please run Command Prompt as Administrator and try again.")
        input("Press Enter to exit...")
        exit(1)
    
    test_arp_spoofing() 