#!/usr/bin/env python3
"""
Web Service Wrapper for ARP Detection
Integrates the existing ARPSpoofingDetector with FastAPI and WebSocket
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from .arp_detector import ARPSpoofingDetector
from .prevention_service import ARPPreventionService

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class ARPDetectionService:
    """Web service wrapper for ARP spoofing detection."""
    
    def __init__(self, config_service, alert_service, websocket_manager):
        self.config_service = config_service
        self.alert_service = alert_service
        self.websocket_manager = websocket_manager
        
        # Detection state
        self.is_monitoring = False
        self.current_interface = None
        self.detection_task = None
        
        # Initialize the detector (will be configured when starting)
        self.detector = None
        
        # Initialize prevention service
        self.prevention_service = ARPPreventionService()
        
        # Real-time data for frontend
        self.live_stats = {
            'total_packets': 0,
            'arp_packets': 0,
            'detected_attacks': 0,
            'last_attack_time': None,
            'current_interface': None,
            'monitoring_status': 'stopped',
            # Prevention statistics
            'prevention_active': False,
            'packets_dropped': 0,
            'arp_entries_corrected': 0,
            'quarantined_ips': 0,
            'rate_limited_ips': 0
        }
        
        # Recent detections for dashboard
        self.recent_detections = []
        self.max_recent_detections = 100
        
        # Add some sample data for testing
        self._add_sample_detections()
        
    def _add_sample_detections(self):
        """Add sample detection data for testing."""
        sample_detections = [
            {
                'id': 1,
                'timestamp': '2025-06-29T20:00:00.000000',
                'src_ip': '192.168.1.100',
                'src_mac': '00:11:22:33:44:55',
                'dst_ip': '192.168.1.1',
                'threat_level': 'HIGH',
                'rule_detection': True,
                'rule_reason': 'Suspicious ARP activity detected',
                'ml_prediction': True,
                'ml_confidence': 0.85
            },
            {
                'id': 2,
                'timestamp': '2025-06-29T20:01:00.000000',
                'src_ip': '192.168.1.101',
                'src_mac': '00:11:22:33:44:66',
                'dst_ip': '192.168.1.1',
                'threat_level': 'MEDIUM',
                'rule_detection': False,
                'rule_reason': 'ML model detected anomaly',
                'ml_prediction': True,
                'ml_confidence': 0.72
            },
            {
                'id': 3,
                'timestamp': '2025-06-29T20:02:00.000000',
                'src_ip': '192.168.1.102',
                'src_mac': '00:11:22:33:44:77',
                'dst_ip': '192.168.1.1',
                'threat_level': 'LOW',
                'rule_detection': True,
                'rule_reason': 'Multiple ARP requests from same source',
                'ml_prediction': False,
                'ml_confidence': 0.45
            }
        ]
        
        self.recent_detections = sample_detections
        logger.info(f"Added {len(sample_detections)} sample detections for testing")
        
    async def start_monitoring(self, interface: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start ARP monitoring on specified interface."""
        try:
            if self.is_monitoring:
                return {"success": False, "message": "Monitoring already active"}
            
            # Get configuration
            if config is None:
                config = await self.config_service.get_detection_config()
            
            # Initialize detector with web integration
            self.detector = ARPSpoofingDetector(
                interface=interface,
                registry_path=config.get('registry_path'),
                model_path=config.get('model_path'),
                log_file=config.get('log_file', 'logs/arp_detector.log'),
                alert_email=config.get('alert_email'),
                webhook_url=config.get('webhook_url'),
                detection_threshold=config.get('detection_threshold', 0.7),
                batch_size=config.get('batch_size', 100),
                max_queue_size=config.get('max_queue_size', 1000),
                auto_registry_addition=config.get('auto_registry_addition', True)
            )
            
            # Override alert methods to integrate with web services
            self.detector.generate_alert = self._web_alert_handler
            self.detector.log_detection = self._web_log_handler
            
            logger.info("Web handlers overridden successfully")
            logger.info(f"Original generate_alert: {self.detector.generate_alert}")
            logger.info(f"Original log_detection: {self.detector.log_detection}")
            
            # Start prevention service if enabled
            prevention_config = config.get('prevention', {})
            logger.info(f"Prevention config: {prevention_config}")
            
            if prevention_config.get('enabled', True):
                logger.info("Starting prevention service...")
                prevention_result = await self.prevention_service.start_prevention(interface, prevention_config)
                if prevention_result['success']:
                    logger.info("Prevention service started successfully")
                    self.live_stats['prevention_active'] = True
                    # Initialize prevention stats to prevent reset
                    await self._sync_prevention_stats()
                else:
                    logger.warning(f"Failed to start prevention service: {prevention_result['message']}")
            else:
                logger.info("Prevention service disabled in config")
            
            # Start monitoring in background task
            self.detection_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_monitoring = True
            self.current_interface = interface
            
            # Update live stats
            self.live_stats.update({
                'monitoring_status': 'running',
                'current_interface': interface,
                'last_attack_time': None
            })
            
            # Broadcast status update
            await self.websocket_manager.broadcast({
                'type': 'monitoring_status',
                'status': 'started',
                'interface': interface,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Started ARP monitoring on interface: {interface}")
            return {"success": True, "message": f"Monitoring started on {interface}"}
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return {"success": False, "message": str(e)}
    
    async def _sync_prevention_stats(self):
        """Synchronize prevention stats to prevent reset during refreshes."""
        try:
            if self.prevention_service and self.prevention_service.is_active:
                prevention_stats = self.prevention_service.get_prevention_stats()
                
                # Ensure we preserve any existing accumulated stats
                current_stats = {
                    'packets_dropped': self.live_stats.get('packets_dropped', 0),
                    'arp_entries_corrected': self.live_stats.get('arp_entries_corrected', 0),
                    'quarantined_ips': self.live_stats.get('quarantined_ips', 0),
                    'rate_limited_ips': self.live_stats.get('rate_limited_ips', 0)
                }
                
                # Use the maximum of current and prevention service stats
                self.live_stats.update({
                    'packets_dropped': max(current_stats['packets_dropped'], prevention_stats['total_packets_dropped']),
                    'arp_entries_corrected': max(current_stats['arp_entries_corrected'], prevention_stats['total_arp_entries_corrected']),
                    'quarantined_ips': max(current_stats['quarantined_ips'], prevention_stats['total_quarantined_ips']),
                    'rate_limited_ips': max(current_stats['rate_limited_ips'], prevention_stats['total_rate_limited_ips']),
                    'prevention_active': prevention_stats['is_active']
                })
                
                logger.info(f"Synced prevention stats: dropped={self.live_stats['packets_dropped']}, corrected={self.live_stats['arp_entries_corrected']}")
                
        except Exception as e:
            logger.error(f"Error syncing prevention stats: {e}")
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop ARP monitoring."""
        try:
            if not self.is_monitoring:
                return {"success": False, "message": "No monitoring active"}
            
            # Stop the detection task
            if self.detection_task:
                self.detection_task.cancel()
                try:
                    await self.detection_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the detector
            if self.detector:
                self.detector.stop_capture()
            
            # Stop prevention service
            if self.prevention_service.is_active:
                await self.prevention_service.stop_prevention()
                self.live_stats['prevention_active'] = False
            
            self.is_monitoring = False
            self.current_interface = None
            
            # Update live stats
            self.live_stats.update({
                'monitoring_status': 'stopped',
                'current_interface': None
            })
            
            # Broadcast status update
            await self.websocket_manager.broadcast({
                'type': 'monitoring_status',
                'status': 'stopped',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("Stopped ARP monitoring")
            return {"success": True, "message": "Monitoring stopped"}
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return {"success": False, "message": str(e)}
    
    async def _monitoring_loop(self):
        """Background task for monitoring."""
        try:
            if self.detector:
                # Start the detector's capture
                self.detector.start_capture()
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await self.stop_monitoring()
    
    def _web_alert_handler(self, detection_result: Dict[str, Any]):
        """Web-integrated alert handler."""
        try:
            # Debug logging to see if this method is being called
            logger.info(f"Web alert handler called for packet: {detection_result.get('src_ip', 'unknown')} -> {detection_result.get('dst_ip', 'unknown')}")
            logger.info(f"Combined threat: {detection_result.get('combined_threat', False)}")
            logger.info(f"Threat level: {detection_result.get('threat_level', 'UNKNOWN')}")
            logger.info(f"Rule detection: {detection_result.get('rule_detection', False)}")
            logger.info(f"ML prediction: {detection_result.get('ml_prediction', None)}")
            
            # Convert numpy types to native Python types
            detection_result = convert_numpy_types(detection_result)
            
            # Only process through prevention service if this is actually a threat
            if detection_result.get('combined_threat', False) and self.prevention_service.is_active:
                logger.info(f"Processing threat through prevention service: {detection_result['src_ip']}")
                # Use thread-safe method instead of async task
                self._process_prevention_sync(detection_result)
            
            # Add to recent detections
            detection_record = {
                'id': len(self.recent_detections) + 1,
                'timestamp': detection_result['timestamp'].isoformat(),
                'src_ip': detection_result['src_ip'],
                'src_mac': detection_result['src_mac'],
                'dst_ip': detection_result['dst_ip'],
                'threat_level': detection_result['threat_level'],
                'rule_detection': detection_result['rule_detection'],
                'rule_reason': detection_result['rule_reason'],
                'ml_prediction': detection_result['ml_prediction'],
                'ml_confidence': float(detection_result['ml_confidence']) if detection_result['ml_confidence'] is not None else None
            }
            
            self.recent_detections.append(detection_record)
            
            # Keep only recent detections
            if len(self.recent_detections) > self.max_recent_detections:
                self.recent_detections.pop(0)
            
            # Update live stats - this is the key fix
            if detection_result.get('combined_threat', False):
                self.live_stats['detected_attacks'] += 1
                self.live_stats['last_attack_time'] = detection_result['timestamp'].isoformat()
                
                logger.info(f"THREAT DETECTED: {detection_result['src_ip']} -> {detection_result['dst_ip']} (Total threats: {self.live_stats['detected_attacks']})")
            
            # Send alert through alert service (thread-safe)
            if detection_result.get('combined_threat', False):
                logger.info(f"Sending alert for threat: {detection_result['src_ip']}")
                alert_created = self.alert_service.send_alert_sync(detection_result)
                
                if alert_created:
                    # Get the latest alert from the alert service
                    latest_alerts = self.alert_service.alert_history
                    if latest_alerts:
                        latest_alert = latest_alerts[-1]  # Get the most recent alert
                        
                        # Broadcast new alert to WebSocket clients
                        self.websocket_manager.broadcast_sync({
                            'type': 'new_alert',
                            'data': latest_alert
                        })
                        
                        logger.info(f"New alert broadcasted: {latest_alert['title']}")
                
                # Broadcast the detection record for dashboard updates
                self.websocket_manager.broadcast_sync({
                    'type': 'attack_detected',
                    'data': detection_record
                })
                
                # Also broadcast updated stats immediately
                self.websocket_manager.broadcast_sync({
                    'type': 'stats_update',
                    'data': self.live_stats.copy()
                })
                
                logger.info(f"Broadcasted threat detection: {detection_record}")
            
        except Exception as e:
            logger.error(f"Error in web alert handler: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_prevention_sync(self, detection_result: Dict[str, Any]):
        """Thread-safe prevention processing."""
        try:
            # Prepare packet data for prevention service
            packet_data = {
                'src_ip': detection_result.get('src_ip'),
                'src_mac': detection_result.get('src_mac'),
                'dst_ip': detection_result.get('dst_ip'),
                'type': 'arp',
                'timestamp': detection_result.get('timestamp'),
                'threat_level': detection_result.get('threat_level'),
                'ml_confidence': detection_result.get('ml_confidence')
            }
            
            # Process through prevention service synchronously
            # We'll use a simple approach that doesn't require async
            if self.prevention_service.is_active:
                # Check if IP was already quarantined before adding
                was_already_quarantined = packet_data['src_ip'] in self.prevention_service.quarantine_list
                
                # Add to quarantine directly
                self.prevention_service._quarantine_ip_sync(
                    packet_data['src_ip'], 
                    packet_data['src_mac'], 
                    f"ARP spoofing detected - {packet_data['threat_level']} threat"
                )
                
                # Update live stats by accumulating (not resetting)
                self.live_stats['packets_dropped'] = self.live_stats.get('packets_dropped', 0) + 1
                
                # Only increment quarantined_ips counter if this is a new quarantine
                if not was_already_quarantined:
                    self.live_stats['quarantined_ips'] = self.live_stats.get('quarantined_ips', 0) + 1
                
                logger.info(f"Prevention action taken: quarantine - ARP spoofing detected")
                logger.info(f"Updated prevention stats: dropped={self.live_stats['packets_dropped']}, quarantined={self.live_stats['quarantined_ips']}")
                
                # Broadcast prevention action
                self.websocket_manager.broadcast_sync({
                    'type': 'prevention_action',
                    'data': {
                        'action': 'quarantine',
                        'reason': 'arp_spoofing_detected',
                        'src_ip': packet_data['src_ip'],
                        'src_mac': packet_data['src_mac'],
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
                # Also broadcast updated stats immediately
                self.websocket_manager.broadcast_sync({
                    'type': 'stats_update',
                    'data': self.live_stats.copy()
                })
                
        except Exception as e:
            logger.error(f"Error processing prevention: {e}")
    
    def _web_log_handler(self, detection_result: Dict[str, Any]):
        """Web-integrated log handler."""
        try:
            # Debug logging to see if this method is being called
            logger.info(f"Web log handler called for packet: {detection_result.get('src_ip', 'unknown')} -> {detection_result.get('dst_ip', 'unknown')}")
            logger.info(f"Combined threat: {detection_result.get('combined_threat', False)}")
            
            # Update packet statistics from detector
            if hasattr(self.detector, 'detection_stats'):
                stats = self.detector.detection_stats
                # Convert numpy types to native Python types
                stats = convert_numpy_types(stats)
                
                # Only update packet counts, not detected_attacks (which is managed by web service)
                self.live_stats.update({
                    'total_packets': int(stats.get('total_packets', 0)),
                    'arp_packets': int(stats.get('arp_packets', 0))
                    # Don't update detected_attacks here - it's managed by _web_alert_handler
                })
                
                logger.info(f"Updated packet stats: total={self.live_stats['total_packets']}, arp={self.live_stats['arp_packets']}")
            
            # Update prevention statistics (but preserve detected_attacks count and accumulated prevention stats)
            if self.prevention_service.is_active:
                # Store current detected_attacks count and current prevention stats
                current_detected_attacks = self.live_stats.get('detected_attacks', 0)
                current_packets_dropped = self.live_stats.get('packets_dropped', 0)
                current_arp_entries_corrected = self.live_stats.get('arp_entries_corrected', 0)
                current_quarantined_ips = self.live_stats.get('quarantined_ips', 0)
                current_rate_limited_ips = self.live_stats.get('rate_limited_ips', 0)
                
                # Get prevention stats synchronously
                prevention_stats = self.prevention_service.get_prevention_stats()
                
                # Use the maximum of current live stats and prevention service stats to prevent reset
                # This ensures we don't lose accumulated stats during refreshes
                self.live_stats.update({
                    'packets_dropped': max(current_packets_dropped, prevention_stats['total_packets_dropped']),
                    'arp_entries_corrected': max(current_arp_entries_corrected, prevention_stats['total_arp_entries_corrected']),
                    'quarantined_ips': max(current_quarantined_ips, prevention_stats['total_quarantined_ips']),
                    'rate_limited_ips': max(current_rate_limited_ips, prevention_stats['total_rate_limited_ips']),
                    'prevention_active': prevention_stats['is_active']
                })
                
                # Restore detected_attacks count
                self.live_stats['detected_attacks'] = current_detected_attacks
                
                logger.info(f"Updated prevention stats: dropped={self.live_stats['packets_dropped']}, corrected={self.live_stats['arp_entries_corrected']}")
            
            # Broadcast stats update periodically (using thread-safe method)
            if self.live_stats['total_packets'] % 100 == 0:  # Every 100 packets
                # Use the websocket manager's thread-safe broadcast method
                self.websocket_manager.broadcast_sync({
                    'type': 'stats_update',
                    'data': self.live_stats.copy()
                })
                logger.info(f"Broadcasted stats update: {self.live_stats}")
                
        except Exception as e:
            logger.error(f"Error in web log handler: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        # Ensure prevention stats are properly synced before returning
        if self.prevention_service and self.prevention_service.is_active:
            await self._sync_prevention_stats()
        
        status = {
            'is_monitoring': self.is_monitoring,
            'current_interface': self.current_interface,
            'live_stats': self.live_stats,
            'recent_detections_count': len(self.recent_detections)
        }
        
        # Add prevention status
        if self.prevention_service:
            prevention_stats = self.prevention_service.get_prevention_stats()
            status['prevention'] = prevention_stats
        
        return status
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        # Ensure prevention stats are properly synced before returning
        if self.prevention_service and self.prevention_service.is_active:
            await self._sync_prevention_stats()
        
        if not self.detector:
            return self.live_stats
        
        stats = self.detector.detection_stats.copy()
        # Convert numpy types to native Python types
        stats = convert_numpy_types(stats)
        
        stats.update({
            'monitoring_status': 'running' if self.is_monitoring else 'stopped',
            'current_interface': self.current_interface,
            'recent_detections': self.recent_detections[-10:]  # Last 10 detections
        })
        
        # Add prevention statistics
        if self.prevention_service:
            prevention_stats = self.prevention_service.get_prevention_stats()
            stats['prevention'] = prevention_stats
        
        return stats
    
    async def get_recent_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent detection records."""
        try:
            logger.info(f"Starting get_recent_detections with limit: {limit}")
            
            # Ensure all data is serializable
            detections = []
            
            # Safely get the recent detections
            recent_detections = getattr(self, 'recent_detections', [])
            logger.info(f"Found {len(recent_detections)} recent detections")
            
            if not isinstance(recent_detections, list):
                logger.warning("recent_detections is not a list, initializing as empty list")
                self.recent_detections = []
                recent_detections = []
            
            # Get the last 'limit' detections
            start_index = max(0, len(recent_detections) - limit)
            recent_slice = recent_detections[start_index:]
            logger.info(f"Processing {len(recent_slice)} detections (from index {start_index})")
            
            for i, detection in enumerate(recent_slice):
                try:
                    logger.debug(f"Processing detection {i+1}: {detection}")
                    
                    # Convert numpy types to native Python types
                    detection = convert_numpy_types(detection)
                    
                    # Create a copy with converted types, with safe defaults
                    serializable_detection = {
                        'id': int(detection.get('id', 0)),
                        'timestamp': str(detection.get('timestamp', datetime.now().isoformat())),
                        'src_ip': str(detection.get('src_ip', '0.0.0.0')),
                        'src_mac': str(detection.get('src_mac', '00:00:00:00:00:00')),
                        'dst_ip': str(detection.get('dst_ip', '0.0.0.0')),
                        'threat_level': str(detection.get('threat_level', 'UNKNOWN')),
                        'rule_detection': bool(detection.get('rule_detection', False)),
                        'rule_reason': str(detection.get('rule_reason', 'No reason provided')),
                        'ml_prediction': detection.get('ml_prediction', None),
                        'ml_confidence': float(detection.get('ml_confidence', 0.0)) if detection.get('ml_confidence') is not None else None
                    }
                    detections.append(serializable_detection)
                    logger.debug(f"Successfully serialized detection {i+1}")
                    
                except Exception as e:
                    logger.error(f"Error serializing detection record {i+1}: {e}")
                    logger.error(f"Problematic detection record: {detection}")
                    # Skip this record and continue with others
                    continue
            
            logger.info(f"Successfully serialized {len(detections)} detection records")
            return detections
            
        except Exception as e:
            logger.error(f"Error in get_recent_detections: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty list instead of raising exception
            return []
    
    async def get_network_interfaces(self) -> List[str]:
        """Get available network interfaces with user-friendly names."""
        try:
            import platform
            import psutil
            
            if platform.system() == "Windows":
                # Use psutil to get user-friendly interface names on Windows
                interfaces = []
                for interface_name, interface_addresses in psutil.net_if_addrs().items():
                    # Skip loopback and virtual interfaces
                    if interface_name.startswith('Loopback') or 'VMware' in interface_name or 'VirtualBox' in interface_name:
                        continue
                    
                    # Check if interface has IPv4 addresses
                    has_ipv4 = any(addr.family == 2 for addr in interface_addresses)  # AF_INET = 2
                    if has_ipv4:
                        interfaces.append(interface_name)
                
                # If psutil fails or returns empty, fallback to Scapy
                if not interfaces:
                    from scapy.all import get_if_list
                    raw_interfaces = get_if_list()
                    # Filter out loopback and format Windows device names
                    for iface in raw_interfaces:
                        if 'Loopback' not in iface and 'NPF_' in iface:
                            # Try to get a more user-friendly name
                            try:
                                # Extract the GUID part and try to get interface info
                                guid = iface.split('NPF_')[1].rstrip('}')
                                interfaces.append(f"Interface {guid[:8]}...")
                            except:
                                interfaces.append(iface)
                
                return interfaces
            else:
                # For Linux/Unix, use Scapy as before
                from scapy.all import get_if_list
                return get_if_list()
                
        except ImportError:
            # Fallback to Scapy if psutil not available
            try:
                from scapy.all import get_if_list
                return get_if_list()
            except Exception as e:
                logger.error(f"Error getting interfaces: {e}")
                return []
        except Exception as e:
            logger.error(f"Error getting interfaces: {e}")
            return []
    
    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update detection configuration."""
        try:
            # Update config service
            await self.config_service.update_detection_config(config)
            
            # If monitoring is active, restart with new config
            if self.is_monitoring and self.current_interface:
                await self.stop_monitoring()
                await asyncio.sleep(1)  # Brief pause
                return await self.start_monitoring(self.current_interface, config)
            
            return {"success": True, "message": "Configuration updated"}
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return {"success": False, "message": str(e)}
    
    # Prevention service methods
    async def get_prevention_stats(self) -> Dict[str, Any]:
        """Get prevention statistics."""
        if self.prevention_service:
            return self.prevention_service.get_prevention_stats()
        return {"is_active": False}
    
    async def get_quarantine_list(self) -> List[Dict[str, Any]]:
        """Get quarantine list."""
        if self.prevention_service:
            return self.prevention_service.get_quarantine_list()
        return []
    
    async def get_rate_limits(self) -> List[Dict[str, Any]]:
        """Get rate limit entries."""
        if self.prevention_service:
            return self.prevention_service.get_rate_limits()
        return []
    
    async def add_legitimate_entry(self, ip: str, mac: str) -> Dict[str, Any]:
        """Add a legitimate IP-MAC mapping."""
        if self.prevention_service:
            return await self.prevention_service.add_legitimate_entry(ip, mac)
        return {"success": False, "message": "Prevention service not available"}
    
    async def remove_quarantine(self, ip: str) -> Dict[str, Any]:
        """Remove an IP from quarantine."""
        if self.prevention_service:
            return await self.prevention_service.remove_quarantine(ip)
        return {"success": False, "message": "Prevention service not available"}
    
    async def clear_prevention_data(self) -> Dict[str, Any]:
        """Clear all prevention data."""
        if self.prevention_service:
            return await self.prevention_service.clear_all_prevention_data()
        return {"success": False, "message": "Prevention service not available"}
    
    async def test_prevention_action(self, ip: str, mac: str, threat_level: str = 'HIGH') -> Dict[str, Any]:
        """Test prevention action with a mock threat."""
        if self.prevention_service:
            return await self.prevention_service.test_prevention_action(ip, mac, threat_level)
        return {"success": False, "message": "Prevention service not available"}
    
    def get_arp_table(self) -> List[Dict[str, Any]]:
        """Get current ARP table entries."""
        if self.prevention_service:
            return self.prevention_service.get_arp_table()
        return [] 