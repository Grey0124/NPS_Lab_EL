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
        
        # Real-time data for frontend
        self.live_stats = {
            'total_packets': 0,
            'arp_packets': 0,
            'detected_attacks': 0,
            'last_attack_time': None,
            'current_interface': None,
            'monitoring_status': 'stopped'
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
            # Convert numpy types to native Python types
            detection_result = convert_numpy_types(detection_result)
            
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
            
            # Update live stats
            self.live_stats['detected_attacks'] += 1
            self.live_stats['last_attack_time'] = detection_result['timestamp'].isoformat()
            
            # Send alert through alert service (thread-safe)
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
            
            # Also broadcast the detection record for dashboard updates
            self.websocket_manager.broadcast_sync({
                'type': 'attack_detected',
                'data': detection_record
            })
            
        except Exception as e:
            logger.error(f"Error in web alert handler: {e}")
    
    def _web_log_handler(self, detection_result: Dict[str, Any]):
        """Web-integrated log handler."""
        try:
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
            
            # Broadcast stats update periodically (using thread-safe method)
            if self.live_stats['total_packets'] % 100 == 0:  # Every 100 packets
                # Use the websocket manager's thread-safe broadcast method
                self.websocket_manager.broadcast_sync({
                    'type': 'stats_update',
                    'data': self.live_stats.copy()
                })
                
        except Exception as e:
            logger.error(f"Error in web log handler: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'current_interface': self.current_interface,
            'live_stats': self.live_stats,
            'recent_detections_count': len(self.recent_detections)
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
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
        """Get available network interfaces."""
        try:
            from scapy.all import get_if_list
            return get_if_list()
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