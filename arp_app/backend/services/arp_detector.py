#!/usr/bin/env python3
"""
Complete ARP Spoofing Detection Tool
Integrates network sniffer, ML model, real-time detection, and alert system.
"""

import argparse
import time
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
import queue
import signal
import socket
import struct
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import joblib
from scapy.all import *
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from .arp_registry import ARPRegistry

class ARPSpoofingDetector:
    """Complete ARP spoofing detection system with ML integration."""
    
    def __init__(self, interface=None, registry_path=None, model_path=None, 
                 log_file=None, alert_email=None, webhook_url=None, 
                 detection_threshold=0.7, batch_size=100, max_queue_size=1000,
                 auto_registry_addition=True):
        
        # Configuration
        self.interface = interface
        self.registry_path = registry_path
        self.model_path = model_path
        self.log_file = log_file
        self.alert_email = alert_email
        self.webhook_url = webhook_url
        self.detection_threshold = detection_threshold
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.auto_registry_addition = auto_registry_addition
        
        # State management
        self.running = False
        self.packet_queue = queue.Queue(maxsize=max_queue_size)
        self.detection_stats = {
            'total_packets': 0,
            'arp_packets': 0,
            'detected_attacks': 0,
            'false_positives': 0,
            'last_alert_time': None,
            'registry_additions': 0
        }
        
        # Network state tracking
        self.arp_cache = {}  # IP -> MAC mapping
        self.mac_history = defaultdict(deque)  # MAC -> recent IPs
        self.arp_history = defaultdict(lambda: {'count': 0, 'suspicious': False, 'last_seen': None})  # IP -> history
        self.suspicious_ips = set()
        
        # ML model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Registry for known IP-MAC pairs
        self.registry_manager = ARPRegistry(registry_path or 'data/registry.yml')
        self.registry = self.registry_manager.entries
        
        # Setup logging
        self.setup_logging()
        
        # Load components (registry auto-created by ARPRegistry)
        self.load_ml_model()
        
        # Alert system
        self.alert_cooldown = 300  # 5 minutes between alerts for same IP
        self.recent_alerts = {}
        
    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_file or 'logs/arp_detector.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ARP Spoofing Detector initialized")
        
    def load_ml_model(self):
        """Load the trained ML model."""
        if not self.model_path or not os.path.exists(self.model_path):
            self.logger.warning(f"No ML model found at path: {self.model_path}, using rule-based detection only")
            return
            
        try:
            self.logger.info(f"Loading ML model from: {self.model_path}")
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            
            self.logger.info(f"Loaded ML model with {len(self.feature_names)} features")
            self.logger.info(f"Model features: {self.feature_names}")
            self.logger.info(f"Model type: {type(self.model).__name__}")
            self.logger.info(f"Detection threshold: {self.detection_threshold}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            self.model = None
            
    def extract_features(self, packet):
        """Extract features from packet for ML prediction."""
        if not self.model:
            return None
            
        try:
            features = {}
            
            # Basic packet features
            features['frame.time_delta'] = 0.0  # Will be calculated in batch
            features['tcp.hdr_len'] = 0.0
            features['tcp.flag_ack'] = 0.0
            features['tcp.flag_psh'] = 0.0
            features['tcp.flag_rst'] = 0.0
            features['tcp.flag_fin'] = 0.0
            features['icmp.type'] = 0.0
            
            # Extract TCP features if present
            if TCP in packet:
                features['tcp.hdr_len'] = packet[TCP].dataofs * 4 if hasattr(packet[TCP], 'dataofs') else 20.0
                features['tcp.flag_ack'] = 1.0 if packet[TCP].flags & 0x10 else 0.0
                features['tcp.flag_psh'] = 1.0 if packet[TCP].flags & 0x08 else 0.0
                features['tcp.flag_rst'] = 1.0 if packet[TCP].flags & 0x04 else 0.0
                features['tcp.flag_fin'] = 1.0 if packet[TCP].flags & 0x01 else 0.0
                
            # Extract ICMP features if present
            if ICMP in packet:
                features['icmp.type'] = float(packet[ICMP].type)
                
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
                
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
            
    def predict_attack(self, features):
        """Predict attack type using ML model."""
        if not self.model or features is None:
            return None, 0.0
            
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get prediction confidence
            confidence = np.max(probabilities)
            
            # Decode prediction
            if self.label_encoder:
                prediction_label = self.label_encoder.inverse_transform([prediction])[0]
            else:
                prediction_label = prediction
                
            # Debug logging for ML predictions
            if confidence > 0.5:  # Only log high-confidence predictions
                self.logger.info(f"ML Prediction: {prediction_label} (confidence: {confidence:.3f})")
                
            return prediction_label, confidence
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return None, 0.0
            
    def check_arp_spoofing_rules(self, packet):
        """Check for ARP spoofing using rule-based detection."""
        if ARP not in packet:
            return False, "Not an ARP packet"
            
        arp = packet[ARP]
        src_ip = arp.psrc
        src_mac = arp.hwsrc
        
        # Check for gratuitous ARP (spoofing indicator)
        if arp.op == 1 and arp.pdst == arp.psrc:  # ARP request to self
            return True, "Gratuitous ARP detected"
            
        # Check if IP-MAC pair is in registry
        if src_ip in self.registry:
            expected_mac = self.registry[src_ip]
            if src_mac.lower() != expected_mac.lower():
                return True, f"MAC mismatch: {src_mac} != {expected_mac}"
                
        # Check for suspicious patterns
        if src_mac in self.mac_history:
            recent_ips = list(self.mac_history[src_mac])
            if src_ip not in recent_ips and len(recent_ips) > 2:
                return True, f"MAC {src_mac} claiming multiple IPs: {recent_ips + [src_ip]}"
                
        # Check for broadcast MAC claiming specific IPs
        if src_mac.lower() in ["ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"]:
            return True, f"Broadcast MAC claiming IP {src_ip}"
            
        # Check for suspicious MAC addresses (fake MACs)
        if src_mac.lower().startswith(("00:11:22", "aa:bb:cc", "de:ad:be")):
            return True, f"Suspicious MAC address: {src_mac}"
            
        return False, "No rule violations"
        
    def process_packet(self, packet):
        """Process a single packet for ARP spoofing detection."""
        try:
            self.detection_stats['total_packets'] += 1
            
            # Check if it's an ARP packet
            if ARP not in packet:
                return
                
            self.detection_stats['arp_packets'] += 1
            arp = packet[ARP]
            
            # Extract packet info
            src_ip = arp.psrc
            src_mac = arp.hwsrc
            dst_ip = arp.pdst
            dst_mac = arp.hwdst
            
            # Log all ARP packets for debugging
            self.logger.info(f"ARP: {src_ip}({src_mac}) -> {dst_ip}({dst_mac}) [op={arp.op}]")
            
            # Update MAC history
            self.mac_history[src_mac].append(src_ip)
            if len(self.mac_history[src_mac]) > 10:
                self.mac_history[src_mac].popleft()
                
            # Rule-based detection
            is_spoofing, rule_reason = self.check_arp_spoofing_rules(packet)
            
            # ML-based detection
            ml_features = self.extract_features(packet)
            ml_prediction, ml_confidence = self.predict_attack(ml_features)

            # If ML model is not loaded, set to None
            if self.model is None:
                ml_confidence = None
                ml_prediction = None

            # Combine detections
            detection_result = {
                'timestamp': datetime.now(),
                'src_ip': src_ip,
                'src_mac': src_mac,
                'dst_ip': dst_ip,
                'dst_mac': dst_mac,
                'arp_op': arp.op,
                'rule_detection': is_spoofing,
                'rule_reason': rule_reason,
                'ml_prediction': ml_prediction,
                'ml_confidence': ml_confidence,
                'combined_threat': False,
                'threat_level': 'LOW'
            }
            
            # Determine combined threat
            ml_positive = (
                self.model is not None and
                ml_prediction is not None and
                ml_prediction != 0 and
                ml_confidence is not None and
                ml_confidence > self.detection_threshold
            )

            if is_spoofing or ml_positive:
                detection_result['combined_threat'] = True
                # Only use ML confidence for threat level if ML is available, else default
                if ml_confidence is not None and ml_confidence > 0.8:
                    detection_result['threat_level'] = 'HIGH'
                else:
                    detection_result['threat_level'] = 'MEDIUM'
                self.detection_stats['detected_attacks'] += 1
                
                # Debug logging for threat detection
                self.logger.warning(f"THREAT DETECTED: {src_ip} -> {dst_ip}")
                self.logger.warning(f"  Rule detection: {is_spoofing} ({rule_reason})")
                self.logger.warning(f"  ML detection: {ml_positive} (prediction: {ml_prediction}, confidence: {ml_confidence})")
                self.logger.warning(f"  Threat level: {detection_result['threat_level']}")
                
                # Generate alert
                self.generate_alert(detection_result)
                
            # Log detection (this will call the web handler for every packet)
            self.log_detection(detection_result)
            
            # Update ARP history for validation
            if ARP in packet:
                # Track ARP history for this IP
                if src_ip in self.arp_history:
                    self.arp_history[src_ip]['count'] += 1
                    self.arp_history[src_ip]['last_seen'] = time.time()
                    if detection_result['combined_threat']:
                        self.arp_history[src_ip]['suspicious'] = True
                else:
                    self.arp_history[src_ip] = {
                        'count': 1,
                        'suspicious': detection_result['combined_threat'],
                        'last_seen': time.time()
                    }
            
            # Automatic registry addition for valid ARP entries
            if ARP in packet and self.auto_registry_addition:
                # Only add to registry if this is NOT a threat (valid ARP mapping)
                if not detection_result['combined_threat']:
                    if self.is_valid_arp_entry(src_ip, src_mac):
                        added = self.registry_manager.add_entry(src_ip, src_mac)
                        if added:
                            self.logger.info(f"✅ AUTO-ADDED to registry: {src_ip} -> {src_mac}")
                            self.detection_stats['registry_additions'] += 1
                        else:
                            self.logger.info(f"ℹ️  Registry entry already exists: {src_ip} -> {src_mac}")
                    else:
                        self.logger.info(f"🚫 Skipped adding to registry (not valid): {src_ip} -> {src_mac}")
                else:
                    self.logger.info(f"🚫 Skipped adding to registry (threat detected): {src_ip} -> {src_mac}")
                
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
            
    def generate_alert(self, detection_result):
        """Generate and send alerts for detected attacks."""
        self.logger.info(f"generate_alert called for threat: {detection_result['src_ip']} -> {detection_result['dst_ip']}")
        
        current_time = time.time()
        src_ip = detection_result['src_ip']
        
        # Check cooldown
        if src_ip in self.recent_alerts:
            if current_time - self.recent_alerts[src_ip] < self.alert_cooldown:
                self.logger.debug(f"Alert cooldown active for {src_ip}")
                return
                
        self.recent_alerts[src_ip] = current_time
        
        # Create alert message
        alert_msg = self.create_alert_message(detection_result)
        
        # Send alerts
        self.send_email_alert(alert_msg)
        self.send_webhook_alert(detection_result)
        
        # Log alert
        self.logger.warning(f"ALERT: {alert_msg}")
        
    def create_alert_message(self, detection_result):
        """Create human-readable alert message."""
        msg = f"""
[ALERT] ARP SPOOFING ATTACK DETECTED

Time: {detection_result['timestamp']}
Source IP: {detection_result['src_ip']}
Source MAC: {detection_result['src_mac']}
Target IP: {detection_result['dst_ip']}
Threat Level: {detection_result['threat_level']}

Detection Methods:
- Rule-based: {'YES' if detection_result['rule_detection'] else 'NO'} ({detection_result['rule_reason']})
- ML-based: {'YES' if detection_result['ml_prediction'] else 'NO'} (Confidence: {detection_result['ml_confidence']:.2f})

Action Required: Investigate network traffic from {detection_result['src_ip']}
        """
        return msg.strip()
        
    def send_email_alert(self, message):
        """Send email alert."""
        if not self.alert_email:
            return
            
        try:
            # This is a placeholder - you'd need to configure SMTP settings
            self.logger.info(f"Email alert would be sent to: {self.alert_email}")
            # Implement actual email sending here
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            
    def send_webhook_alert(self, detection_result):
        """Send webhook alert."""
        if not self.webhook_url:
            return
            
        try:
            payload = {
                'timestamp': detection_result['timestamp'].isoformat(),
                'src_ip': detection_result['src_ip'],
                'src_mac': detection_result['src_mac'],
                'threat_level': detection_result['threat_level'],
                'ml_confidence': detection_result['ml_confidence'],
                'rule_reason': detection_result['rule_reason']
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                self.logger.info("Webhook alert sent successfully")
            else:
                self.logger.warning(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            
    def log_detection(self, detection_result):
        """Log detection results."""
        self.logger.debug(f"log_detection called for packet: {detection_result['src_ip']} -> {detection_result['dst_ip']}")
        
        if detection_result['combined_threat']:
            self.logger.warning(f"[THREAT] DETECTED: {detection_result['src_ip']} -> {detection_result['dst_ip']} "
                              f"(ML: {detection_result['ml_prediction']}, "
                              f"Rule: {detection_result['rule_detection']})")
            self.logger.warning(f"   Rule reason: {detection_result['rule_reason']}")
            self.logger.warning(f"   ML confidence: {detection_result['ml_confidence']:.3f}")
        else:
            self.logger.debug(f"Normal: {detection_result['src_ip']} -> {detection_result['dst_ip']}")
            
    def packet_callback(self, packet):
        """Callback for packet capture."""
        try:
            if not self.packet_queue.full():
                self.packet_queue.put(packet)
            else:
                self.logger.warning("Packet queue full, dropping packet")
        except Exception as e:
            self.logger.error(f"Error in packet callback: {e}")
            
    def process_packet_queue(self):
        """Process packets from the queue."""
        while self.running:
            try:
                # Process batch of packets
                packets = []
                for _ in range(self.batch_size):
                    try:
                        packet = self.packet_queue.get(timeout=1)
                        packets.append(packet)
                    except queue.Empty:
                        break
                        
                # Process packets
                for packet in packets:
                    self.process_packet(packet)
                    
            except Exception as e:
                self.logger.error(f"Error processing packet queue: {e}")
                
    def start_capture(self):
        """Start packet capture."""
        if not self.interface:
            self.logger.error("No interface specified")
            return False
            
        try:
            self.logger.info(f"Starting packet capture on interface: {self.interface}")
            
            # Start packet processing thread
            self.running = True
            process_thread = threading.Thread(target=self.process_packet_queue)
            process_thread.daemon = True
            process_thread.start()
            
            # Start packet capture in a separate thread (non-blocking)
            capture_thread = threading.Thread(target=self._capture_packets)
            capture_thread.daemon = True
            capture_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    def _capture_packets(self):
        """Non-blocking packet capture thread."""
        try:
            # Use non-blocking sniff with timeout
            sniff(iface=self.interface, prn=self.packet_callback, store=0, stop_filter=lambda p: not self.running)
        except Exception as e:
            self.logger.error(f"Error in packet capture thread: {e}")
        finally:
            self.logger.info("Packet capture thread stopped")
        
    def stop_capture(self):
        """Stop packet capture."""
        self.logger.info("Stopping packet capture...")
        self.running = False
        
    def print_stats(self):
        """Print detection statistics."""
        stats = self.detection_stats
        print(f"\n{'='*50}")
        print("ARP SPOOFING DETECTION STATISTICS")
        print(f"{'='*50}")
        print(f"Total Packets: {stats['total_packets']:,}")
        print(f"ARP Packets: {stats['arp_packets']:,}")
        print(f"Detected Attacks: {stats['detected_attacks']:,}")
        print(f"Detection Rate: {stats['detected_attacks']/max(stats['arp_packets'], 1)*100:.2f}%")
        print(f"Last Alert: {stats['last_alert_time'] or 'None'}")
        print(f"Registry Additions: {stats['registry_additions']:,}")
        print(f"Auto Registry Addition: {'Enabled' if self.auto_registry_addition else 'Disabled'}")
        print(f"Registry Entries: {len(self.registry_manager.entries)}")
        print(f"{'='*50}")
    
    def get_registry_stats(self):
        """Get registry-related statistics."""
        return {
            'auto_registry_addition': self.auto_registry_addition,
            'registry_additions': self.detection_stats.get('registry_additions', 0),
            'total_registry_entries': len(self.registry_manager.entries),
            'arp_history_size': len(self.arp_history)
        }

    def is_valid_arp_entry(self, ip, mac):
        """
        Determine if an ARP entry is valid for automatic addition to registry.
        Enhanced logic to be more intelligent about what constitutes a valid entry.
        """
        # Basic validation
        if not ip or not mac:
            return False
            
        # Check if IP is in suspicious list
        if ip in self.suspicious_ips:
            self.logger.debug(f"Skipping suspicious IP for registry: {ip}")
            return False
            
        # Check if IP is in private ranges (more likely to be valid)
        try:
            ip_parts = ip.split('.')
            if len(ip_parts) == 4:
                first_octet = int(ip_parts[0])
                second_octet = int(ip_parts[1])
                
                # Common private IP ranges - be more inclusive
                if (first_octet == 10 or  # 10.0.0.0/8
                    (first_octet == 172 and 16 <= second_octet <= 31) or  # 172.16.0.0/12
                    (first_octet == 192 and second_octet == 168)):  # 192.168.0.0/16 (all subnets)
                    return True
                    
                # Localhost
                if ip == "127.0.0.1":
                    return True
                    
                # Link-local addresses
                if first_octet == 169 and second_octet == 254:
                    return True
                    
                # Special addresses
                if ip == "0.0.0.0":
                    return True
                    
        except (ValueError, IndexError):
            pass
            
        # For public IPs, be more conservative
        # Only add if we've seen consistent behavior
        if ip in self.arp_history:
            # If we've seen this IP multiple times without issues, consider it valid
            if self.arp_history[ip]['count'] >= 3 and not self.arp_history[ip]['suspicious']:
                return True
                
        # Default: don't add public IPs automatically unless we have more context
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nShutting down ARP Spoofing Detector...")
    if detector:
        detector.stop_capture()
    sys.exit(0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ARP Spoofing Detection Tool')
    parser.add_argument('-i', '--interface', help='Network interface to monitor')
    parser.add_argument('-r', '--registry', help='Path to IP-MAC registry YAML file')
    parser.add_argument('-m', '--model', default='ML/models/realistic_rf_model.joblib', 
                       help='Path to trained ML model')
    parser.add_argument('-l', '--log', default='logs/arp_detector.log', 
                       help='Log file path')
    parser.add_argument('-e', '--email', help='Email address for alerts')
    parser.add_argument('-w', '--webhook', help='Webhook URL for alerts')
    parser.add_argument('-t', '--threshold', type=float, default=0.7, 
                       help='ML detection threshold (0.0-1.0)')
    parser.add_argument('--list-interfaces', action='store_true', 
                       help='List available network interfaces')
    
    args = parser.parse_args()
    
    # List interfaces if requested
    if args.list_interfaces:
        print("Available network interfaces:")
        for iface in get_if_list():
            print(f"  - {iface}")
        return
    
    # Check if interface is provided for detection
    if not args.interface:
        parser.error("Interface (-i/--interface) is required for detection mode")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create detector
    global detector
    detector = ARPSpoofingDetector(
        interface=args.interface,
        registry_path=args.registry,
        model_path=args.model,
        log_file=args.log,
        alert_email=args.email,
        webhook_url=args.webhook,
        detection_threshold=args.threshold
    )
    
    # Start detection
    print("Starting ARP Spoofing Detection Tool...")
    print(f"Interface: {args.interface}")
    print(f"ML Model: {args.model}")
    print(f"Detection Threshold: {args.threshold}")
    print("Press Ctrl+C to stop\n")
    
    try:
        detector.start_capture()
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        detector.stop_capture()
        detector.print_stats()

if __name__ == "__main__":
    detector = None
    main() 