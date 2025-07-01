#!/usr/bin/env python3
"""
Test script to verify ARP detection system is working properly
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from services.arp_detector_service import ARPDetectionService
from services.config_service import ConfigService
from services.alert_service import AlertService
from websocket.manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_detection_system():
    """Test the detection system."""
    try:
        logger.info("Initializing test detection system...")
        
        # Initialize services
        config_service = ConfigService()
        alert_service = AlertService()
        websocket_manager = WebSocketManager()
        
        # Start WebSocket queue processor
        await websocket_manager.start_queue_processor()
        
        # Initialize ARP detection service
        arp_service = ARPDetectionService(
            config_service=config_service,
            alert_service=alert_service,
            websocket_manager=websocket_manager
        )
        
        logger.info("Detection system initialized successfully")
        
        # Test manual threat detection first
        logger.info("Testing manual threat detection...")
        test_detection_result = {
            'timestamp': datetime.now(),
            'src_ip': '192.168.1.100',
            'src_mac': '00:11:22:33:44:55',
            'dst_ip': '192.168.1.1',
            'dst_mac': 'ff:ff:ff:ff:ff:ff',
            'arp_op': 1,
            'rule_detection': True,
            'rule_reason': 'Test threat detection',
            'ml_prediction': 1,
            'ml_confidence': 0.85,
            'combined_threat': True,
            'threat_level': 'HIGH'
        }
        
        # Call the web alert handler directly
        arp_service._web_alert_handler(test_detection_result)
        
        # Check if the threat was recorded
        status = await arp_service.get_status()
        logger.info(f"Status after manual test: {status}")
        
        # Get available interfaces
        interfaces = await arp_service.get_network_interfaces()
        logger.info(f"Available interfaces: {interfaces}")
        
        if not interfaces:
            logger.error("No network interfaces available")
            return
        
        # Test with first available interface
        test_interface = interfaces[0]
        logger.info(f"Testing with interface: {test_interface}")
        
        # Start monitoring
        logger.info("Starting monitoring...")
        result = await arp_service.start_monitoring(test_interface)
        logger.info(f"Start monitoring result: {result}")
        
        if result['success']:
            logger.info("Monitoring started successfully")
            
            # Wait a bit for the system to initialize
            await asyncio.sleep(5)
            
            # Check status
            status = await arp_service.get_status()
            logger.info(f"Monitoring status: {status}")
            
            # Wait for some packets to be processed
            logger.info("Waiting for packets to be processed...")
            await asyncio.sleep(30)
            
            # Check status again
            status = await arp_service.get_status()
            logger.info(f"Updated monitoring status: {status}")
            
            # Get recent detections
            detections = await arp_service.get_recent_detections(10)
            logger.info(f"Recent detections: {len(detections)} found")
            
            # Stop monitoring
            logger.info("Stopping monitoring...")
            stop_result = await arp_service.stop_monitoring()
            logger.info(f"Stop monitoring result: {stop_result}")
            
        else:
            logger.error(f"Failed to start monitoring: {result['message']}")
        
        # Cleanup
        await websocket_manager.shutdown()
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_detection_system()) 