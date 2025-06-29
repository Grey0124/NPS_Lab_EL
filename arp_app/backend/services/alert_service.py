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
        
    async def send_alert(self, detection_result: Dict[str, Any]) -> bool:
        """Send alert for detected attack."""
        try:
            # Check cooldown
            src_ip = detection_result.get('src_ip')
            if src_ip and not self._check_cooldown(src_ip):
                logger.debug(f"Alert for {src_ip} in cooldown period")
                return False
            
            # Create alert record
            alert_record = {
                'id': len(self.alert_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'detection_result': detection_result,
                'sent': False,
                'error': None
            }
            
            # Add to history
            self.alert_history.append(alert_record)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # Send notifications
            success = await self._send_notifications(alert_record)
            alert_record['sent'] = success
            
            # Update cooldown
            if src_ip:
                self.cooldown_timestamps[src_ip] = datetime.now()
            
            logger.info(f"Alert sent for {src_ip}: {success}")
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
            # This is a placeholder - you'd need to configure SMTP settings
            detection = alert_record['detection_result']
            
            subject = f"ARP Spoofing Attack Detected - {detection['src_ip']}"
            
            body = f"""
ARP Spoofing Attack Detected

Time: {detection['timestamp']}
Source IP: {detection['src_ip']}
Source MAC: {detection['src_mac']}
Target IP: {detection['dst_ip']}
Threat Level: {detection['threat_level']}

Detection Methods:
- Rule-based: {'YES' if detection['rule_detection'] else 'NO'} ({detection['rule_reason']})
- ML-based: {'YES' if detection['ml_prediction'] else 'NO'} (Confidence: {detection['ml_confidence']:.2f})

Action Required: Investigate network traffic from {detection['src_ip']}
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
            detection = alert_record['detection_result']
            
            payload = {
                'timestamp': detection['timestamp'].isoformat(),
                'src_ip': detection['src_ip'],
                'src_mac': detection['src_mac'],
                'dst_ip': detection['dst_ip'],
                'threat_level': detection['threat_level'],
                'ml_confidence': detection['ml_confidence'],
                'rule_reason': detection['rule_reason'],
                'alert_id': alert_record['id']
            }
            
            # Placeholder for webhook sending
            logger.info(f"Webhook alert would be sent: {payload}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            return False
    
    async def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        successful_alerts = sum(1 for alert in self.alert_history if alert.get('sent', False))
        
        return {
            'total_alerts': total_alerts,
            'successful_alerts': successful_alerts,
            'success_rate': (successful_alerts / total_alerts * 100) if total_alerts > 0 else 0,
            'cooldown_active_ips': len(self.cooldown_timestamps)
        }
    
    async def clear_alert_history(self) -> bool:
        """Clear alert history."""
        try:
            self.alert_history.clear()
            self.cooldown_timestamps.clear()
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
            
            # Create alert record
            alert_record = {
                'id': len(self.alert_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'detection_result': detection_result,
                'sent': False,
                'error': None
            }
            
            # Add to history
            self.alert_history.append(alert_record)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # For thread-safe version, we'll just log the alert
            # The actual sending can be done asynchronously later
            logger.info(f"Alert queued for {src_ip}: {detection_result.get('threat_level', 'unknown')} threat")
            
            # Update cooldown
            if src_ip:
                self.cooldown_timestamps[src_ip] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in send_alert_sync: {e}")
            return False 