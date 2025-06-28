#!/usr/bin/env python3
"""
ARP Packet Sniffer Module

This module provides functionality to capture and analyze ARP packets on a specified network interface.
It includes features for packet capture, analysis, and logging of ARP requests and replies.
"""

import sys
import os
import platform
import logging
import logging.handlers
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, List
from scapy.all import (
    ARP,
    Ether,
    sniff,
    get_if_list,
    get_if_hwaddr,
    get_if_addr,
    conf,
    sendp
)
from scapy.error import Scapy_Exception

# Windows-specific configuration
if platform.system() == "Windows":
    # Force use of Npcap/WinPcap
    conf.use_pcap = True
    # Check for administrator privileges
    if not os.environ.get("ADMIN"):
        print("Warning: Running without administrator privileges. Some features may not work.")
        print("Please run the script as administrator for full functionality.")

# Initialize logger (will be configured in ARPSniffer._setup_logging)
logger = logging.getLogger(__name__)

class ARPSniffer:
    """ARP packet sniffer class for capturing and analyzing ARP traffic."""

    def __init__(self, interface: str = "eth0", registry_file: str = "registry.yml", log_file: Optional[str] = None):
        """
        Initialize the ARP sniffer.

        Args:
            interface (str): Network interface to capture packets on
            registry_file (str): Path to the registry YAML file
            log_file (Optional[str]): Path to log file for mismatches (default: None, use stdout)
        """
        self.interface = interface
        self.registry_file = registry_file
        self.log_file = log_file
        self.arp_cache: Dict[str, Dict[str, Any]] = {}
        self.registry: Dict[str, Any] = {}
        self.registry_devices: Dict[str, Dict[str, Any]] = {}
        self.registry_settings: Dict[str, Any] = {}
        self._setup_logging()
        self._load_registry()
        self._validate_interface()
        self._check_platform_requirements()

    def _setup_logging(self) -> None:
        """Setup logging configuration with optional rolling file handler."""
        # Clear any existing handlers
        logger = logging.getLogger(__name__)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set log level
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler for general info
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for mismatches if log_file is specified
        if self.log_file:
            try:
                # Create rolling file handler (10MB max size, keep 5 backup files)
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.WARNING)  # Only log warnings and errors
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"Logging mismatches to: {self.log_file}")
            except Exception as e:
                logger.error(f"Failed to setup log file {self.log_file}: {str(e)}")
                logger.info("Continuing with console logging only")

    def _check_platform_requirements(self) -> None:
        """Check platform-specific requirements."""
        if platform.system() == "Windows":
            try:
                # Try to get interface info to verify Npcap is working
                get_if_hwaddr(self.interface)
            except Scapy_Exception as e:
                if "Npcap" in str(e) or "WinPcap" in str(e):
                    logger.error("Npcap/WinPcap not found or not properly installed.")
                    logger.error("Please install Npcap with 'WinPcap API-compatible Mode' enabled.")
                    sys.exit(1)
                raise

    def _load_registry(self) -> None:
        """Load the registry YAML file containing known IP→MAC mappings."""
        try:
            if not os.path.exists(self.registry_file):
                logger.warning(f"Registry file {self.registry_file} not found. Creating empty registry.")
                self.registry = {
                    'devices': {},
                    'network': {},
                    'settings': {
                        'strict_mode': False,
                        'log_unknown_devices': True,
                        'auto_update': False,
                        'check_interval': 300
                    }
                }
                return

            with open(self.registry_file, 'r', encoding='utf-8') as file:
                self.registry = yaml.safe_load(file)
                
            # Extract devices and settings
            self.registry_devices = self.registry.get('devices', {})
            self.registry_settings = self.registry.get('settings', {})
            
            logger.info(f"Loaded registry with {len(self.registry_devices)} known devices")
            logger.info(f"Registry settings: strict_mode={self.registry_settings.get('strict_mode', False)}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing registry file {self.registry_file}: {str(e)}")
            logger.info("Using empty registry")
            self.registry = {'devices': {}, 'settings': {}}
            self.registry_devices = {}
            self.registry_settings = {}
        except Exception as e:
            logger.error(f"Error loading registry file {self.registry_file}: {str(e)}")
            logger.info("Using empty registry")
            self.registry = {'devices': {}, 'settings': {}}
            self.registry_devices = {}
            self.registry_settings = {}

    def _check_registry_compliance(self, ip: str, mac: str) -> Dict[str, Any]:
        """
        Check if an IP→MAC mapping complies with the registry.

        Args:
            ip (str): IP address to check
            mac (str): MAC address to check

        Returns:
            Dict[str, Any]: Compliance check results
        """
        result = {
            'compliant': True,
            'in_registry': False,
            'device_info': None,
            'warning': None
        }

        # Check if IP exists in registry
        if ip in self.registry_devices:
            result['in_registry'] = True
            device_info = self.registry_devices[ip]
            result['device_info'] = device_info
            
            # Check if MAC matches
            if device_info.get('mac', '').lower() != mac.lower():
                result['compliant'] = False
                result['warning'] = f"MAC mismatch for IP {ip}"
                
        else:
            # IP not in registry
            if self.registry_settings.get('strict_mode', False):
                result['compliant'] = False
                result['warning'] = f"Unknown IP {ip} in strict mode"
            elif self.registry_settings.get('log_unknown_devices', True):
                result['warning'] = f"Unknown device IP {ip} ({mac})"

        return result

    def _get_available_interfaces(self) -> List[str]:
        """
        Get list of available network interfaces.

        Returns:
            List[str]: List of interface names
        """
        return get_if_list()

    def _format_interface_list(self, interfaces: List[str]) -> str:
        """
        Format the list of interfaces for display.

        Args:
            interfaces (List[str]): List of interface names

        Returns:
            str: Formatted string of interfaces
        """
        if platform.system() == "Windows":
            # For Windows, show a more user-friendly format with interface details
            formatted = []
            for i, iface in enumerate(interfaces, 1):
                try:
                    mac = get_if_hwaddr(iface)
                    ip = get_if_addr(iface)
                    formatted.append(f"{i}. {iface}\n   MAC: {mac}\n   IP: {ip}")
                except Scapy_Exception:
                    formatted.append(f"{i}. {iface}")
            return "\n".join(formatted)
        return "\n".join(interfaces)

    def _validate_interface(self) -> None:
        """Validate that the specified interface exists and is available."""
        available_interfaces = self._get_available_interfaces()
        
        # If interface is not found, show available interfaces and exit
        if self.interface not in available_interfaces:
            logger.error(f"Interface {self.interface} not found.")
            logger.info("Available interfaces:")
            logger.info(self._format_interface_list(available_interfaces))
            logger.info("\nTo use a specific interface, run with -i option followed by the interface name.")
            sys.exit(1)

        try:
            # Test if we can get the MAC address of the interface
            get_if_hwaddr(self.interface)
        except Scapy_Exception as e:
            logger.error(f"Error accessing interface {self.interface}: {str(e)}")
            sys.exit(1)

    def _get_interface_info(self) -> Dict[str, str]:
        """
        Get interface information including MAC and IP address.

        Returns:
            Dict[str, str]: Dictionary containing interface information
        """
        try:
            return {
                'mac': get_if_hwaddr(self.interface),
                'ip': get_if_addr(self.interface)
            }
        except Scapy_Exception as e:
            logger.error(f"Error getting interface info: {str(e)}")
            return {'mac': 'unknown', 'ip': 'unknown'}

    def _process_arp_packet(self, packet: Ether) -> None:
        """
        Process an ARP packet and extract relevant information.

        Args:
            packet (Ether): The captured packet
        """
        if not packet.haslayer(ARP):
            return

        arp = packet[ARP]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Extract packet information
        packet_info = {
            'timestamp': timestamp,
            'op': 'request' if arp.op == 1 else 'reply',
            'src_mac': arp.hwsrc,
            'src_ip': arp.psrc,
            'dst_mac': arp.hwdst,
            'dst_ip': arp.pdst
        }

        # Check registry compliance for source IP
        src_compliance = self._check_registry_compliance(arp.psrc, arp.hwsrc)
        if not src_compliance['compliant']:
            logger.warning(
                f"REGISTRY VIOLATION - Source: {arp.psrc} ({arp.hwsrc})\n"
                f"    Warning: {src_compliance['warning']}\n"
                f"    Device Info: {src_compliance['device_info']}"
            )
        elif src_compliance['warning']:
            logger.info(f"Registry notice - Source: {src_compliance['warning']}")

        # Check registry compliance for destination IP (if not broadcast)
        if arp.pdst != "0.0.0.0":
            dst_compliance = self._check_registry_compliance(arp.pdst, arp.hwdst)
            if not dst_compliance['compliant']:
                logger.warning(
                    f"REGISTRY VIOLATION - Destination: {arp.pdst} ({arp.hwdst})\n"
                    f"    Warning: {dst_compliance['warning']}\n"
                    f"    Device Info: {dst_compliance['device_info']}"
                )
            elif dst_compliance['warning']:
                logger.info(f"Registry notice - Destination: {dst_compliance['warning']}")

        # Check for potential ARP spoofing (existing functionality)
        if arp.psrc in self.arp_cache:
            cached_mac = self.arp_cache[arp.psrc]['mac']
            if cached_mac != arp.hwsrc:
                logger.warning(
                    f"Potential ARP spoofing detected!\n"
                    f"IP: {arp.psrc}\n"
                    f"Old MAC: {cached_mac}\n"
                    f"New MAC: {arp.hwsrc}"
                )

        # Update ARP cache
        self.arp_cache[arp.psrc] = {
            'mac': arp.hwsrc,
            'last_seen': timestamp
        }

        # Log the packet
        self._log_packet(packet_info)

    def _log_packet(self, packet_info: Dict[str, Any]) -> None:
        """
        Log ARP packet information.

        Args:
            packet_info (Dict[str, Any]): Dictionary containing packet information
        """
        log_message = (
            f"[{packet_info['timestamp']}] ARP {packet_info['op'].upper()}\n"
            f"    Source: {packet_info['src_ip']} ({packet_info['src_mac']})\n"
            f"    Target: {packet_info['dst_ip']} ({packet_info['dst_mac']})"
        )
        logger.info(log_message)

    def start_sniffing(self, count: Optional[int] = None) -> None:
        """
        Start capturing ARP packets on the specified interface.

        Args:
            count (Optional[int]): Number of packets to capture. If None, capture indefinitely.
        """
        interface_info = self._get_interface_info()
        logger.info(f"Starting ARP sniffer on interface {self.interface}")
        logger.info(f"Interface MAC: {interface_info['mac']}")
        logger.info(f"Interface IP: {interface_info['ip']}")
        logger.info("Press Ctrl+C to stop")

        try:
            sniff(
                iface=self.interface,
                filter="arp",
                prn=self._process_arp_packet,
                store=0,
                count=count
            )
        except KeyboardInterrupt:
            logger.info("\nStopping ARP sniffer...")
        except Scapy_Exception as e:
            logger.error(f"Error during packet capture: {str(e)}")
            sys.exit(1)

def list_interfaces() -> None:
    """List all available network interfaces."""
    sniffer = ARPSniffer(interface="eth0", log_file=None)  # Temporary instance just to use its methods
    interfaces = sniffer._get_available_interfaces()
    logger.info("Available network interfaces:")
    logger.info(sniffer._format_interface_list(interfaces))

def main():
    """Main function to run the ARP sniffer."""
    import argparse

    parser = argparse.ArgumentParser(description='ARP Packet Sniffer with Registry Checking')
    parser.add_argument(
        '-i', '--interface',
        default='eth0',
        help='Network interface to capture packets on (default: eth0)'
    )
    parser.add_argument(
        '-c', '--count',
        type=int,
        help='Number of packets to capture (default: capture indefinitely)'
    )
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List available network interfaces'
    )
    parser.add_argument(
        '-r', '--registry',
        default='registry.yml',
        help='Path to registry YAML file (default: registry.yml)'
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file for mismatches (default: stdout only)'
    )
    args = parser.parse_args()

    if args.list:
        list_interfaces()
        return

    sniffer = ARPSniffer(
        interface=args.interface, 
        registry_file=args.registry,
        log_file=args.log_file
    )
    sniffer.start_sniffing(count=args.count)

if __name__ == "__main__":
    main() 