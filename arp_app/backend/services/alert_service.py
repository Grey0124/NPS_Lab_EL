#!/usr/bin/env python3
"""
Alert Service for ARP Detection Web Application
"""

import asyncio
import json
import logging
import smtplib
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertService:
    """Service for handling alerts and notifications."""
    
    def __init__(self):
        self.alert_history = []
        self.max_history = 1000
        self.cooldown_timestamps = {}  # IP -> last alert time
        self.cooldown_period = 300  # 5 minutes
        self.alert_counter = 0
        
        # Clear any existing test alerts
        self._clear_test_alerts()
        
        # Add some sample alerts for testing
        # self._add_sample_alerts()
        
    def _clear_test_alerts(self):
        """Clear any existing test alerts from the history."""
        # Remove any alerts with test IPs or old timestamps
        test_ips = ['192.168.1.100', '192.168.1.50', '192.168.1.75']
        original_count = len(self.alert_history)
        
        self.alert_history = [
            alert for alert in self.alert_history 
            if not (
                alert.get('sourceIP') in test_ips or
                alert.get('timestamp', '').startswith('2024-01-15')
            )
        ]
        
        removed_count = original_count - len(self.alert_history)
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} test alerts from history")
        
    def _add_sample_alerts(self):
        """Add sample alerts for testing purposes."""
        # Commented out to remove test alerts
        pass
        # sample_alerts = [
        #     {
        #         'id': '1',
        #         'timestamp': '2024-01-15T10:30:00Z',
        #         'type': 'arp_spoofing',
        #         'severity': 'critical',
        #         'title': 'Critical ARP Spoofing Attack Detected',
        #         'description': 'Multiple ARP spoofing attempts detected from 192.168.1.100 targeting gateway 192.168.1.1. Immediate action required.',
        #         'sourceIP': '192.168.1.100',
        #         'targetIP': '192.168.1.1',
        #         'status': 'new',
        #         'acknowledgedBy': None,
        #         'resolvedAt': None
        #     },
        #     {
        #         'id': '2',
        #         'timestamp': '2024-01-15T09:15:00Z',
        #         'type': 'suspicious_activity',
        #         'severity': 'high',
        #         'title': 'Suspicious Network Activity Detected',
        #         'description': 'Unusual ARP traffic patterns detected from 192.168.1.50. ML model confidence: 0.85. Investigation recommended.',
        #         'sourceIP': '192.168.1.50',
        #         'targetIP': '192.168.1.1',
        #         'status': 'acknowledged',
        #         'acknowledgedBy': 'admin',
        #         'resolvedAt': None
        #     },
        #     {
        #         'id': '3',
        #         'timestamp': '2024-01-15T08:45:00Z',
        #         'type': 'arp_spoofing',
        #         'severity': 'medium',
        #         'title': 'Potential ARP Spoofing Attempt',
        #         'description': 'Possible ARP spoofing attempt from 192.168.1.75. Rule-based detection triggered. Monitor closely.',
        #         'sourceIP': '192.168.1.75',
        #         'targetIP': '192.168.1.1',
        #         'status': 'resolved',
        #         'acknowledgedBy': 'admin',
        #         'resolvedAt': '2024-01-15T09:00:00Z'
        #     },
        #     {
        #         'id': '4',
        #         'timestamp': '2024-01-15T07:30:00Z',
        #         'type': 'system_error',
        #         'severity': 'low',
        #         'title': 'Network Interface Configuration Change',
        #         'description': 'Network interface eth0 configuration changed. This may affect monitoring capabilities.',
        #         'sourceIP': None,
        #         'targetIP': None,
        #         'status': 'dismissed',
        #         'acknowledgedBy': 'system',
        #         'resolvedAt': None
        #     }
        # ]
        # 
        # for alert in sample_alerts:
        #     self.alert_history.append(alert)
        #     self.alert_counter = max(self.alert_counter, int(alert['id']))
        # 
        # logger.info(f"Added {len(sample_alerts)} sample alerts for testing")
    
    def _convert_detection_to_alert(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert detection result to Alert format expected by frontend."""
        self.alert_counter += 1
        
        # Determine severity based on threat level and ML confidence
        threat_level = detection_result.get('threat_level', 'MEDIUM').lower()
        ml_confidence = detection_result.get('ml_confidence', 0)
        
        if threat_level == 'high' or ml_confidence > 0.8:
            severity = 'critical'
        elif threat_level == 'medium' or ml_confidence > 0.6:
            severity = 'high'
        elif ml_confidence > 0.4:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Determine alert type based on detection method
        if detection_result.get('rule_detection'):
            alert_type = 'arp_spoofing'
        elif detection_result.get('ml_prediction'):
            alert_type = 'suspicious_activity'
        else:
            alert_type = 'system_error'
        
        # Create title and description
        src_ip = detection_result.get('src_ip', 'Unknown')
        dst_ip = detection_result.get('dst_ip', 'Unknown')
        rule_reason = detection_result.get('rule_reason', 'Unknown reason')
        
        title = f"ARP Spoofing Attack Detected from {src_ip}"
        description = f"Potential ARP spoofing attack detected from {src_ip} targeting {dst_ip}. "
        
        if detection_result.get('rule_detection'):
            description += f"Rule-based detection: {rule_reason}. "
        
        if detection_result.get('ml_prediction') and ml_confidence:
            description += f"ML confidence: {ml_confidence:.2f}. "
        
        description += "Immediate investigation recommended."
        
        return {
            'id': str(self.alert_counter),
            'timestamp': detection_result.get('timestamp', datetime.now()).isoformat(),
            'type': alert_type,
            'severity': severity,
            'title': title,
            'description': description,
            'sourceIP': src_ip,
            'targetIP': dst_ip,
            'status': 'new',
            'acknowledgedBy': None,
            'resolvedAt': None
        }
    
    async def send_alert(self, detection_result: Dict[str, Any]) -> bool:
        """Send alert for detected attack."""
        try:
            # Check cooldown
            src_ip = detection_result.get('src_ip')
            if src_ip and not self._check_cooldown(src_ip):
                logger.debug(f"Alert for {src_ip} in cooldown period")
                return False
            
            # Convert detection result to alert format
            alert_record = self._convert_detection_to_alert(detection_result)
            
            # Add to history
            self.alert_history.append(alert_record)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # Send notifications
            success = await self._send_notifications(alert_record)
            
            # Update cooldown
            if src_ip:
                self.cooldown_timestamps[src_ip] = datetime.now()
            
            logger.info(f"Alert created for {src_ip}: {alert_record['title']}")
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def _check_cooldown(self, ip: str) -> bool:
        """Check if IP is in cooldown period."""
        if ip not in self.cooldown_timestamps:
            return True
        
        time_diff = (datetime.now() - self.cooldown_timestamps[ip]).total_seconds()
        return time_diff > self.cooldown_period
    
    async def _send_notifications(self, alert_record: Dict[str, Any]) -> bool:
        """Send all configured notifications."""
        try:
            # Get alert configuration (this would come from config service)
            # For now, using placeholder config
            config = {
                'email_enabled': False,
                'email_recipients': [],
                'webhook_enabled': False,
                'webhook_url': None
            }
            
            success = True
            
            # Send email if enabled
            if config.get('email_enabled') and config.get('email_recipients'):
                email_success = await self._send_email_alert(alert_record, config)
                success = success and email_success
            
            # Send webhook if enabled
            if config.get('webhook_enabled') and config.get('webhook_url'):
                webhook_success = await self._send_webhook_alert(alert_record, config)
                success = success and webhook_success
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return False
    
    async def _send_email_alert(self, alert_record: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Send email alert."""
        try:
            subject = f"Security Alert: {alert_record['title']}"
            
            body = f"""
Security Alert Detected

Time: {alert_record['timestamp']}
Type: {alert_record['type']}
Severity: {alert_record['severity']}
Title: {alert_record['title']}
Description: {alert_record['description']}

Source IP: {alert_record.get('sourceIP', 'Unknown')}
Target IP: {alert_record.get('targetIP', 'Unknown')}

Action Required: Investigate this security incident immediately.
            """
            
            # Placeholder for email sending
            logger.info(f"Email alert would be sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False
    
    async def _send_webhook_alert(self, alert_record: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Send webhook alert."""
        try:
            payload = {
                'timestamp': alert_record['timestamp'],
                'alert_id': alert_record['id'],
                'type': alert_record['type'],
                'severity': alert_record['severity'],
                'title': alert_record['title'],
                'description': alert_record['description'],
                'sourceIP': alert_record.get('sourceIP'),
                'targetIP': alert_record.get('targetIP')
            }
            
            # Placeholder for webhook sending
            logger.info(f"Webhook alert would be sent: {payload}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False
    
    async def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history in the format expected by frontend."""
        # Return alerts in reverse chronological order (newest first)
        alerts = self.alert_history[-limit:][::-1]
        return alerts
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics in the format expected by frontend."""
        total_alerts = len(self.alert_history)
        
        # Count by status
        new_count = sum(1 for alert in self.alert_history if alert.get('status') == 'new')
        acknowledged_count = sum(1 for alert in self.alert_history if alert.get('status') == 'acknowledged')
        resolved_count = sum(1 for alert in self.alert_history if alert.get('status') == 'resolved')
        dismissed_count = sum(1 for alert in self.alert_history if alert.get('status') == 'dismissed')
        
        # Count by severity
        critical_count = sum(1 for alert in self.alert_history if alert.get('severity') == 'critical')
        high_count = sum(1 for alert in self.alert_history if alert.get('severity') == 'high')
        medium_count = sum(1 for alert in self.alert_history if alert.get('severity') == 'medium')
        low_count = sum(1 for alert in self.alert_history if alert.get('severity') == 'low')
        
        return {
            'total': total_alerts,
            'new': new_count,
            'acknowledged': acknowledged_count,
            'resolved': resolved_count,
            'dismissed': dismissed_count,
            'critical': critical_count,
            'high': high_count,
            'medium': medium_count,
            'low': low_count
        }
    
    async def clear_alert_history(self) -> bool:
        """Clear alert history."""
        try:
            self.alert_history.clear()
            self.cooldown_timestamps.clear()
            self.alert_counter = 0
            logger.info("Alert history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing alert history: {e}")
            return False
    
    async def update_cooldown_period(self, seconds: int) -> bool:
        """Update cooldown period."""
        try:
            self.cooldown_period = max(0, seconds)
            logger.info(f"Cooldown period updated to {seconds} seconds")
            return True
        except Exception as e:
            logger.error(f"Error updating cooldown period: {e}")
            return False
    
    def send_alert_sync(self, detection_result: Dict[str, Any]) -> bool:
        """Thread-safe version of send_alert that can be called from any thread."""
        try:
            # Check cooldown
            src_ip = detection_result.get('src_ip')
            if src_ip and not self._check_cooldown(src_ip):
                logger.debug(f"Alert for {src_ip} in cooldown period")
                return False
            
            # Convert detection result to alert format
            alert_record = self._convert_detection_to_alert(detection_result)
            
            # Add to history
            self.alert_history.append(alert_record)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # For thread-safe version, we'll just log the alert
            # The actual sending can be done asynchronously later
            logger.info(f"Alert queued for {src_ip}: {alert_record['title']}")
            
            # Update cooldown
            if src_ip:
                self.cooldown_timestamps[src_ip] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in send_alert_sync: {e}")
            return False 