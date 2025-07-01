#!/usr/bin/env python3
"""
ARP Spoofing Prevention Service
Handles packet dropping, ARP table management, and active response
"""

import asyncio
import logging
import platform
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class PreventionStats:
    """Statistics for prevention activities."""
    total_packets_dropped: int = 0
    total_arp_entries_corrected: int = 0
    total_quarantined_ips: int = 0
    total_rate_limited_ips: int = 0
    last_prevention_time: Optional[datetime] = None
    prevention_duration: float = 0.0

@dataclass
class QuarantineEntry:
    """Represents a quarantined IP address."""
    ip: str
    mac: str
    reason: str
    quarantined_at: datetime
    expires_at: datetime
    attempts: int = 0

@dataclass
class RateLimitEntry:
    """Represents a rate-limited IP address."""
    ip: str
    mac: str
    first_seen: datetime
    last_seen: datetime
    packet_count: int = 0
    blocked_until: Optional[datetime] = None

class ARPPreventionService:
    """ARP spoofing prevention service with packet dropping and active response."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_active = False
        self.current_interface = None
        
        # Prevention statistics
        self.stats = PreventionStats()
        
        # Quarantine management
        self.quarantine_list: Dict[str, QuarantineEntry] = {}
        self.quarantine_duration = self.config.get('quarantine_duration', 300)  # 5 minutes default
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitEntry] = {}
        self.rate_limit_threshold = self.config.get('rate_limit_threshold', 100)  # packets per minute
        self.rate_limit_duration = self.config.get('rate_limit_duration', 60)  # 1 minute default
        
        # ARP table management
        self.arp_table_cache: Dict[str, str] = {}  # IP -> MAC mapping
        self.legitimate_arp_entries: Dict[str, str] = {}  # Known good IP->MAC mappings
        
        # Thread safety
        self.lock = threading.Lock()
        self.packet_queue = queue.Queue()
        
        # Background tasks
        self.cleanup_task = None
        self.stats_update_task = None
        
        # Platform detection
        self.is_windows = platform.system() == "Windows"
        
        logger.info(f"ARP Prevention Service initialized for {platform.system()}")
    
    async def start_prevention(self, interface: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start the prevention service on the specified interface."""
        try:
            if self.is_active:
                return {"success": False, "message": "Prevention already active"}
            
            # Preserve existing stats if any
            existing_stats = {
                'total_packets_dropped': self.stats.total_packets_dropped,
                'total_arp_entries_corrected': self.stats.total_arp_entries_corrected,
                'total_quarantined_ips': self.stats.total_quarantined_ips,
                'total_rate_limited_ips': self.stats.total_rate_limited_ips,
                'last_prevention_time': self.stats.last_prevention_time
            }
            
            # Update configuration
            if config:
                self.config.update(config)
                self.quarantine_duration = config.get('quarantine_duration', 300)
                self.rate_limit_threshold = config.get('rate_limit_threshold', 100)
                self.rate_limit_duration = config.get('rate_limit_duration', 60)
            
            self.current_interface = interface
            self.is_active = True
            
            # Restore preserved stats
            self.stats.total_packets_dropped = existing_stats['total_packets_dropped']
            self.stats.total_arp_entries_corrected = existing_stats['total_arp_entries_corrected']
            self.stats.total_quarantined_ips = existing_stats['total_quarantined_ips']
            self.stats.total_rate_limited_ips = existing_stats['total_rate_limited_ips']
            self.stats.last_prevention_time = existing_stats['last_prevention_time']
            
            # Initialize ARP table cache
            await self._initialize_arp_table()
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.stats_update_task = asyncio.create_task(self._stats_update_loop())
            
            logger.info(f"ARP Prevention started on interface: {interface} (stats preserved: dropped={self.stats.total_packets_dropped})")
            return {"success": True, "message": f"Prevention started on {interface}"}
            
        except Exception as e:
            logger.error(f"Failed to start prevention: {e}")
            return {"success": False, "message": str(e)}
    
    async def stop_prevention(self) -> Dict[str, Any]:
        """Stop the prevention service."""
        try:
            if not self.is_active:
                return {"success": False, "message": "No prevention active"}
            
            # Preserve stats before stopping
            preserved_stats = {
                'total_packets_dropped': self.stats.total_packets_dropped,
                'total_arp_entries_corrected': self.stats.total_arp_entries_corrected,
                'total_quarantined_ips': self.stats.total_quarantined_ips,
                'total_rate_limited_ips': self.stats.total_rate_limited_ips,
                'last_prevention_time': self.stats.last_prevention_time
            }
            
            self.is_active = False
            self.current_interface = None
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.stats_update_task:
                self.stats_update_task.cancel()
                try:
                    await self.stats_update_task
                except asyncio.CancelledError:
                    pass
            
            # Clear all prevention data but preserve stats
            with self.lock:
                self.quarantine_list.clear()
                self.rate_limits.clear()
                self.arp_table_cache.clear()
                # Restore preserved stats
                self.stats.total_packets_dropped = preserved_stats['total_packets_dropped']
                self.stats.total_arp_entries_corrected = preserved_stats['total_arp_entries_corrected']
                self.stats.total_quarantined_ips = preserved_stats['total_quarantined_ips']
                self.stats.total_rate_limited_ips = preserved_stats['total_rate_limited_ips']
                self.stats.last_prevention_time = preserved_stats['last_prevention_time']
            
            logger.info("ARP Prevention stopped (stats preserved)")
            return {"success": True, "message": "Prevention stopped"}
            
        except Exception as e:
            logger.error(f"Failed to stop prevention: {e}")
            return {"success": False, "message": str(e)}
    
    async def process_packet(self, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a packet and apply prevention measures."""
        try:
            if not self.is_active:
                return {"action": "none", "reason": "prevention_not_active"}
            
            src_ip = packet_data.get('src_ip')
            src_mac = packet_data.get('src_mac')
            dst_ip = packet_data.get('dst_ip')
            packet_type = packet_data.get('type', 'arp')
            
            if not src_ip or not src_mac:
                return {"action": "none", "reason": "invalid_packet_data"}
            
            # Check quarantine first
            if src_ip in self.quarantine_list:
                return await self._handle_quarantined_packet(src_ip, src_mac, packet_data)
            
            # Check rate limiting
            if await self._is_rate_limited(src_ip, src_mac):
                return await self._handle_rate_limited_packet(src_ip, src_mac, packet_data)
            
            # Check ARP table consistency
            if packet_type == 'arp':
                return await self._handle_arp_packet(src_ip, src_mac, dst_ip, packet_data)
            
            return {"action": "none", "reason": "packet_allowed"}
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            return {"action": "none", "reason": "error_processing"}
    
    async def _handle_quarantined_packet(self, src_ip: str, src_mac: str, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle packets from quarantined IPs."""
        quarantine_entry = self.quarantine_list[src_ip]
        
        # Note: attempts are now handled in the quarantine methods
        # Only extend quarantine if attempts continue beyond threshold
        if quarantine_entry.attempts > 5:
            quarantine_entry.expires_at = datetime.now() + timedelta(seconds=self.quarantine_duration)
        
        self.stats.total_packets_dropped += 1
        self.stats.last_prevention_time = datetime.now()
        
        return {
            "action": "drop",
            "reason": "quarantined_ip",
            "quarantine_info": {
                "reason": quarantine_entry.reason,
                "quarantined_at": quarantine_entry.quarantined_at.isoformat(),
                "expires_at": quarantine_entry.expires_at.isoformat(),
                "attempts": quarantine_entry.attempts
            }
        }
    
    async def _is_rate_limited(self, src_ip: str, src_mac: str) -> bool:
        """Check if an IP is rate limited."""
        now = datetime.now()
        
        if src_ip not in self.rate_limits:
            self.rate_limits[src_ip] = RateLimitEntry(
                ip=src_ip,
                mac=src_mac,
                first_seen=now,
                last_seen=now,
                packet_count=1
            )
            return False
        
        entry = self.rate_limits[src_ip]
        
        # Check if still blocked
        if entry.blocked_until and now < entry.blocked_until:
            return True
        
        # Reset if time window has passed
        if (now - entry.first_seen).total_seconds() > 60:
            entry.first_seen = now
            entry.packet_count = 1
            entry.blocked_until = None
            return False
        
        # Update packet count
        entry.packet_count += 1
        entry.last_seen = now
        
        # Check if threshold exceeded
        if entry.packet_count > self.rate_limit_threshold:
            entry.blocked_until = now + timedelta(seconds=self.rate_limit_duration)
            self.stats.total_rate_limited_ips += 1
            return True
        
        return False
    
    async def _handle_rate_limited_packet(self, src_ip: str, src_mac: str, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rate-limited packets."""
        entry = self.rate_limits[src_ip]
        self.stats.total_packets_dropped += 1
        self.stats.last_prevention_time = datetime.now()
        
        return {
            "action": "drop",
            "reason": "rate_limited",
            "rate_limit_info": {
                "packet_count": entry.packet_count,
                "threshold": self.rate_limit_threshold,
                "blocked_until": entry.blocked_until.isoformat() if entry.blocked_until else None
            }
        }
    
    async def _handle_arp_packet(self, src_ip: str, src_mac: str, dst_ip: str, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ARP packets and check for spoofing."""
        # Check if this IP-MAC mapping is legitimate
        if src_ip in self.legitimate_arp_entries:
            if self.legitimate_arp_entries[src_ip] != src_mac:
                # Potential ARP spoofing detected
                return await self._handle_arp_spoofing(src_ip, src_mac, packet_data)
        
        # Check for suspicious patterns
        if await self._is_suspicious_arp_activity(src_ip, src_mac, packet_data):
            return await self._handle_arp_spoofing(src_ip, src_mac, packet_data)
        
        # Update ARP table cache
        self.arp_table_cache[src_ip] = src_mac
        
        return {"action": "none", "reason": "arp_packet_allowed"}
    
    async def _is_suspicious_arp_activity(self, src_ip: str, src_mac: str, packet_data: Dict[str, Any]) -> bool:
        """Check for suspicious ARP activity patterns."""
        # Check threat level from detection service
        threat_level = packet_data.get('threat_level', 'LOW')
        ml_confidence = packet_data.get('ml_confidence', 0.0)
        
        # High threat level or high ML confidence indicates suspicious activity
        if threat_level == 'HIGH' or (ml_confidence and ml_confidence > 0.7):
            return True
        
        # Check for rapid ARP requests from same source
        now = datetime.now()
        if src_ip in self.rate_limits:
            entry = self.rate_limits[src_ip]
            time_diff = (now - entry.last_seen).total_seconds()
            if time_diff < 1 and entry.packet_count > 10:  # More than 10 ARP requests in 1 second
                return True
        
        return False
    
    async def _handle_arp_spoofing(self, src_ip: str, src_mac: str, packet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detected ARP spoofing."""
        # Add to quarantine with longer duration for ARP spoofing
        quarantine_reason = "ARP spoofing detected"
        if packet_data.get('threat_level') == 'HIGH':
            quarantine_reason += " (High threat level)"
        elif packet_data.get('ml_confidence', 0) > 0.8:
            quarantine_reason += f" (ML confidence: {packet_data.get('ml_confidence', 0):.2f})"
        
        await self._quarantine_ip(src_ip, src_mac, quarantine_reason)
        
        # Correct ARP table if we have a legitimate entry
        if src_ip in self.legitimate_arp_entries:
            await self._correct_arp_table(src_ip, self.legitimate_arp_entries[src_ip])
            corrected_mac = self.legitimate_arp_entries[src_ip]
        else:
            # If no legitimate entry, try to get from system ARP table
            try:
                if self.is_windows:
                    result = subprocess.run(['arp', '-a'], capture_output=True, text=True, check=True)
                    for line in result.stdout.split('\n'):
                        if src_ip in line and 'dynamic' in line.lower():
                            parts = line.split()
                            if len(parts) >= 2 and parts[0] == src_ip:
                                corrected_mac = parts[1]
                                await self._correct_arp_table(src_ip, corrected_mac)
                                break
                    else:
                        corrected_mac = None
                else:
                    result = subprocess.run(['ip', 'neigh', 'show'], capture_output=True, text=True, check=True)
                    for line in result.stdout.split('\n'):
                        if src_ip in line:
                            parts = line.split()
                            if len(parts) >= 4 and parts[0] == src_ip:
                                corrected_mac = parts[4]
                                await self._correct_arp_table(src_ip, corrected_mac)
                                break
                    else:
                        corrected_mac = None
            except Exception as e:
                logger.error(f"Failed to get system ARP entry for {src_ip}: {e}")
                corrected_mac = None
        
        self.stats.total_packets_dropped += 1
        self.stats.last_prevention_time = datetime.now()
        
        return {
            "action": "drop_and_correct",
            "reason": "arp_spoofing",
            "corrected_mac": corrected_mac,
            "quarantine_duration": self.quarantine_duration,
            "threat_level": packet_data.get('threat_level', 'MEDIUM'),
            "ml_confidence": packet_data.get('ml_confidence', 0.0)
        }
    
    async def _quarantine_ip(self, ip: str, mac: str, reason: str):
        """Add an IP to quarantine."""
        with self.lock:
            if ip in self.quarantine_list:
                # IP already quarantined - increment attempts and extend expiration
                existing_entry = self.quarantine_list[ip]
                existing_entry.attempts += 1
                existing_entry.expires_at = datetime.now() + timedelta(seconds=self.quarantine_duration)
                logger.info(f"Updated quarantine for {ip} ({mac}) - attempts: {existing_entry.attempts}")
            else:
                # New quarantine entry
                self.quarantine_list[ip] = QuarantineEntry(
                    ip=ip,
                    mac=mac,
                    reason=reason,
                    quarantined_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=self.quarantine_duration),
                    attempts=1  # Start with 1 attempt since this is the first quarantine
                )
                self.stats.total_quarantined_ips += 1
                logger.info(f"Quarantined IP {ip} ({mac}) for reason: {reason}")
    
    def _quarantine_ip_sync(self, ip: str, mac: str, reason: str):
        """Synchronous version of quarantine for thread-safe calls."""
        with self.lock:
            if ip in self.quarantine_list:
                # IP already quarantined - increment attempts and extend expiration
                existing_entry = self.quarantine_list[ip]
                existing_entry.attempts += 1
                existing_entry.expires_at = datetime.now() + timedelta(seconds=self.quarantine_duration)
                logger.info(f"Updated quarantine for {ip} ({mac}) - attempts: {existing_entry.attempts}")
            else:
                # New quarantine entry
                self.quarantine_list[ip] = QuarantineEntry(
                    ip=ip,
                    mac=mac,
                    reason=reason,
                    quarantined_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=self.quarantine_duration),
                    attempts=1  # Start with 1 attempt since this is the first quarantine
                )
                self.stats.total_quarantined_ips += 1
                logger.info(f"Quarantined IP {ip} ({mac}) for reason: {reason}")
    
    async def _correct_arp_table(self, ip: str, correct_mac: str):
        """Correct the ARP table entry."""
        try:
            if self.is_windows:
                # Windows: Use arp command
                subprocess.run([
                    'arp', '-s', ip, correct_mac
                ], capture_output=True, check=True)
            else:
                # Linux: Use ip neigh command
                subprocess.run([
                    'ip', 'neigh', 'replace', ip, 'lladdr', correct_mac, 'dev', self.current_interface
                ], capture_output=True, check=True)
            
            self.stats.total_arp_entries_corrected += 1
            logger.info(f"Corrected ARP table: {ip} -> {correct_mac}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to correct ARP table: {e}")
        except Exception as e:
            logger.error(f"Error correcting ARP table: {e}")
    
    async def _initialize_arp_table(self):
        """Initialize the ARP table cache."""
        try:
            if self.is_windows:
                # Windows: Parse arp -a output
                result = subprocess.run(['arp', '-a'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if 'dynamic' in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            ip = parts[0]
                            mac = parts[1]
                            self.arp_table_cache[ip] = mac
                            self.legitimate_arp_entries[ip] = mac
            else:
                # Linux: Parse ip neigh show output
                result = subprocess.run(['ip', 'neigh', 'show'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            ip = parts[0]
                            mac = parts[4]
                            self.arp_table_cache[ip] = mac
                            self.legitimate_arp_entries[ip] = mac
            
            logger.info(f"Initialized ARP table with {len(self.arp_table_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to initialize ARP table: {e}")
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while self.is_active:
            try:
                now = datetime.now()
                
                # Clean up expired quarantine entries
                with self.lock:
                    expired_quarantine = [
                        ip for ip, entry in self.quarantine_list.items()
                        if now > entry.expires_at
                    ]
                    for ip in expired_quarantine:
                        del self.quarantine_list[ip]
                        logger.info(f"Removed {ip} from quarantine")
                
                # Clean up old rate limit entries
                with self.lock:
                    old_rate_limits = [
                        ip for ip, entry in self.rate_limits.items()
                        if (now - entry.last_seen).total_seconds() > 300  # 5 minutes
                    ]
                    for ip in old_rate_limits:
                        del self.rate_limits[ip]
                
                await asyncio.sleep(30)  # Clean up every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _stats_update_loop(self):
        """Background task to update statistics."""
        while self.is_active:
            try:
                # Update prevention duration
                if self.stats.last_prevention_time:
                    self.stats.prevention_duration = (
                        datetime.now() - self.stats.last_prevention_time
                    ).total_seconds()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats update loop: {e}")
                await asyncio.sleep(5)
    
    def get_prevention_stats(self) -> Dict[str, Any]:
        """Get current prevention statistics."""
        with self.lock:
            # Ensure stats are not reset by checking if they're being properly accumulated
            stats_data = {
                "is_active": self.is_active,
                "current_interface": self.current_interface,
                "total_packets_dropped": self.stats.total_packets_dropped,
                "total_arp_entries_corrected": self.stats.total_arp_entries_corrected,
                "total_quarantined_ips": self.stats.total_quarantined_ips,
                "total_rate_limited_ips": self.stats.total_rate_limited_ips,
                "last_prevention_time": self.stats.last_prevention_time.isoformat() if self.stats.last_prevention_time else None,
                "prevention_duration": self.stats.prevention_duration,
                "quarantine_count": len(self.quarantine_list),
                "rate_limit_count": len(self.rate_limits),
                "arp_table_size": len(self.arp_table_cache),
                "legitimate_entries": len(self.legitimate_arp_entries)
            }
            
            # Log the stats to help debug any reset issues
            logger.debug(f"Prevention stats: dropped={self.stats.total_packets_dropped}, corrected={self.stats.total_arp_entries_corrected}, quarantined={self.stats.total_quarantined_ips}")
            
            return stats_data
    
    def get_quarantine_list(self) -> List[Dict[str, Any]]:
        """Get current quarantine list."""
        with self.lock:
            return [
                {
                    "ip": entry.ip,
                    "mac": entry.mac,
                    "reason": entry.reason,
                    "quarantined_at": entry.quarantined_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat(),
                    "attempts": entry.attempts
                }
                for entry in self.quarantine_list.values()
            ]
    
    def get_rate_limits(self) -> List[Dict[str, Any]]:
        """Get current rate limit entries."""
        with self.lock:
            return [
                {
                    "ip": entry.ip,
                    "mac": entry.mac,
                    "first_seen": entry.first_seen.isoformat(),
                    "last_seen": entry.last_seen.isoformat(),
                    "packet_count": entry.packet_count,
                    "blocked_until": entry.blocked_until.isoformat() if entry.blocked_until else None
                }
                for entry in self.rate_limits.values()
            ]
    
    async def add_legitimate_entry(self, ip: str, mac: str) -> Dict[str, Any]:
        """Add a legitimate IP-MAC mapping."""
        try:
            with self.lock:
                self.legitimate_arp_entries[ip] = mac
                self.arp_table_cache[ip] = mac
            
            logger.info(f"Added legitimate entry: {ip} -> {mac}")
            return {"success": True, "message": f"Added {ip} -> {mac}"}
            
        except Exception as e:
            logger.error(f"Failed to add legitimate entry: {e}")
            return {"success": False, "message": str(e)}
    
    async def remove_quarantine(self, ip: str) -> Dict[str, Any]:
        """Remove an IP from quarantine."""
        try:
            with self.lock:
                if ip in self.quarantine_list:
                    del self.quarantine_list[ip]
                    logger.info(f"Removed {ip} from quarantine")
                    return {"success": True, "message": f"Removed {ip} from quarantine"}
                else:
                    return {"success": False, "message": f"{ip} not found in quarantine"}
                    
        except Exception as e:
            logger.error(f"Failed to remove from quarantine: {e}")
            return {"success": False, "message": str(e)}
    
    async def clear_all_prevention_data(self) -> Dict[str, Any]:
        """Clear all prevention data."""
        try:
            with self.lock:
                self.quarantine_list.clear()
                self.rate_limits.clear()
                self.arp_table_cache.clear()
                self.legitimate_arp_entries.clear()
            
            # Reset statistics
            self.stats = PreventionStats()
            
            logger.info("Cleared all prevention data")
            return {"success": True, "message": "All prevention data cleared"}
            
        except Exception as e:
            logger.error(f"Failed to clear prevention data: {e}")
            return {"success": False, "message": str(e)}
    
    async def test_prevention_action(self, ip: str, mac: str, threat_level: str = 'HIGH') -> Dict[str, Any]:
        """Test prevention action with a mock threat."""
        try:
            # Create a mock packet data
            packet_data = {
                'src_ip': ip,
                'src_mac': mac,
                'dst_ip': '192.168.1.1',
                'type': 'arp',
                'timestamp': datetime.now(),
                'threat_level': threat_level,
                'ml_confidence': 0.85
            }
            
            # Process through prevention
            result = await self.process_packet(packet_data)
            
            return {
                "success": True,
                "message": f"Test prevention action completed",
                "action": result['action'],
                "reason": result['reason']
            }
            
        except Exception as e:
            logger.error(f"Failed to test prevention action: {e}")
            return {"success": False, "message": str(e)}
    
    def get_arp_table(self) -> List[Dict[str, Any]]:
        """Get current ARP table entries."""
        try:
            arp_entries = []
            seen_entries = set()  # Track unique IP-MAC combinations
            current_interface = None
            
            # Get current system ARP table
            if self.is_windows:
                # Windows: Parse arp -a output
                result = subprocess.run(['arp', '-a'], capture_output=True, text=True, check=True)
                lines = result.stdout.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is an interface header line
                    if line.startswith('Interface:'):
                        # Extract interface information
                        # Format: "Interface: 192.168.59.1 --- 0xa"
                        interface_parts = line.split('---')
                        if len(interface_parts) >= 1:
                            interface_info = interface_parts[0].replace('Interface:', '').strip()
                            # Extract IP address from interface info
                            interface_ip = interface_info.split()[0] if interface_info else None
                            current_interface = interface_ip
                        continue
                    
                    # Check if this is a data line with IP and MAC
                    if 'dynamic' in line.lower() or 'static' in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            ip = parts[0]
                            mac = parts[1]
                            entry_type = 'dynamic' if 'dynamic' in line.lower() else 'static'
                            
                            # Create unique identifier to avoid duplicates
                            entry_key = f"{ip}-{mac}"
                            if entry_key not in seen_entries:
                                seen_entries.add(entry_key)
                                arp_entries.append({
                                    'ip': ip,
                                    'mac': mac,
                                    'type': entry_type,
                                    'interface': current_interface,
                                    'is_legitimate': ip in self.legitimate_arp_entries and self.legitimate_arp_entries[ip] == mac
                                })
            else:
                # Linux: Parse ip neigh show output
                result = subprocess.run(['ip', 'neigh', 'show'], capture_output=True, text=True, check=True)
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            ip = parts[0]
                            mac = parts[4]
                            entry_type = 'dynamic'  # Linux doesn't distinguish in this output
                            
                            # Create unique identifier to avoid duplicates
                            entry_key = f"{ip}-{mac}"
                            if entry_key not in seen_entries:
                                seen_entries.add(entry_key)
                                arp_entries.append({
                                    'ip': ip,
                                    'mac': mac,
                                    'type': entry_type,
                                    'interface': self.current_interface,
                                    'is_legitimate': ip in self.legitimate_arp_entries and self.legitimate_arp_entries[ip] == mac
                                })
            
            logger.info(f"Retrieved {len(arp_entries)} ARP table entries with interfaces")
            return arp_entries
            
        except Exception as e:
            logger.error(f"Failed to get ARP table: {e}")
            # Return cached entries as fallback (also deduplicated)
            seen_entries = set()
            arp_entries = []
            for ip, mac in self.arp_table_cache.items():
                entry_key = f"{ip}-{mac}"
                if entry_key not in seen_entries:
                    seen_entries.add(entry_key)
                    arp_entries.append({
                        'ip': ip,
                        'mac': mac,
                        'type': 'cached',
                        'interface': self.current_interface,
                        'is_legitimate': ip in self.legitimate_arp_entries and self.legitimate_arp_entries[ip] == mac
                    })
            return arp_entries 