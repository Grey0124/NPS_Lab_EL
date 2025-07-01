#!/usr/bin/env python3
"""
Test script to directly test the backend's get_network_interfaces method
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'arp_app', 'backend'))

from services.arp_detector_service import ARPDetectionService
import asyncio

async def test_interfaces():
    """Test the get_network_interfaces method directly."""
    # Create a mock service (we don't need the full dependencies for this test)
    class MockConfigService:
        async def get_detection_config(self):
            return {}
    
    class MockAlertService:
        def send_alert_sync(self, detection_result):
            return True
        @property
        def alert_history(self):
            return []
    
    class MockWebSocketManager:
        async def broadcast(self, message):
            pass
        def broadcast_sync(self, message):
            pass
    
    # Create the service with mock dependencies
    service = ARPDetectionService(
        config_service=MockConfigService(),
        alert_service=MockAlertService(),
        websocket_manager=MockWebSocketManager()
    )
    
    # Test the method
    print("Testing get_network_interfaces method...")
    interfaces = await service.get_network_interfaces()
    print(f"Interfaces returned: {interfaces}")
    
    return interfaces

if __name__ == "__main__":
    interfaces = asyncio.run(test_interfaces())
    print(f"\nFinal result: {interfaces}") 