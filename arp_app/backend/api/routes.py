#!/usr/bin/env python3
"""
API Routes for ARP Detection Web Application
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

# Import services (these will be injected)
from services.arp_detector_service import ARPDetectionService
from services.config_service import ConfigService
from services.alert_service import AlertService
from websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class MonitoringRequest(BaseModel):
    interface: str
    config: Optional[Dict[str, Any]] = None

class ConfigUpdateRequest(BaseModel):
    detection: Optional[Dict[str, Any]] = None
    alerts: Optional[Dict[str, Any]] = None

class AlertConfigRequest(BaseModel):
    email_enabled: bool = False
    email_recipients: List[str] = []
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    notification_cooldown: int = 300

# Dependency injection (these will be set by main.py)
arp_service: Optional[ARPDetectionService] = None
config_service: Optional[ConfigService] = None
alert_service: Optional[AlertService] = None
websocket_manager: Optional[WebSocketManager] = None

def get_arp_service() -> ARPDetectionService:
    if arp_service is None:
        raise HTTPException(status_code=503, detail="ARP service not available")
    return arp_service

def get_config_service() -> ConfigService:
    if config_service is None:
        raise HTTPException(status_code=503, detail="Config service not available")
    return config_service

def get_alert_service() -> AlertService:
    if alert_service is None:
        raise HTTPException(status_code=503, detail="Alert service not available")
    return alert_service

def get_websocket_manager() -> WebSocketManager:
    if websocket_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")
    return websocket_manager

# Monitoring endpoints
@router.post("/monitoring/start")
async def start_monitoring(
    request: MonitoringRequest,
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Start ARP monitoring on specified interface."""
    try:
        result = await arp_service.start_monitoring(request.interface, request.config)
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitoring/stop")
async def stop_monitoring(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Stop ARP monitoring."""
    try:
        result = await arp_service.stop_monitoring()
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/status")
async def get_monitoring_status(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Get current monitoring status."""
    try:
        status = await arp_service.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/interfaces")
async def get_network_interfaces(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Get available network interfaces."""
    try:
        interfaces = await arp_service.get_network_interfaces()
        return {"interfaces": interfaces}
    except Exception as e:
        logger.error(f"Error getting interfaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoints
@router.get("/statistics")
async def get_statistics(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Get detailed detection statistics."""
    try:
        stats = await arp_service.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detections")
async def get_recent_detections(
    limit: int = 50,
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Get recent detection records."""
    try:
        logger.info(f"Getting recent detections with limit: {limit}")
        
        # Check if arp_service is properly initialized
        if arp_service is None:
            logger.error("ARP service is not initialized")
            return {"detections": [], "count": 0, "error": "Service not initialized"}
        
        detections = await arp_service.get_recent_detections(limit)
        
        # Ensure detections is a list
        if not isinstance(detections, list):
            logger.warning(f"Detections is not a list: {type(detections)}")
            detections = []
        
        logger.info(f"Retrieved {len(detections)} detections")
        return {"detections": detections, "count": len(detections)}
        
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty response instead of raising HTTPException
        return {"detections": [], "count": 0, "error": str(e)}

# Configuration endpoints
@router.get("/config")
async def get_configuration(
    config_service: ConfigService = Depends(get_config_service)
):
    """Get current configuration."""
    try:
        config = await config_service.get_full_config()
        return config
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config")
async def update_configuration(
    request: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service),
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Update configuration."""
    try:
        success = True
        
        if request.detection:
            success &= await config_service.update_detection_config(request.detection)
        
        if request.alerts:
            success &= await config_service.update_alert_config(request.alerts)
        
        if success:
            # If monitoring is active, restart with new config
            if arp_service.is_monitoring and arp_service.current_interface:
                await arp_service.stop_monitoring()
                await arp_service.start_monitoring(arp_service.current_interface)
            
            return {"status": "success", "message": "Configuration updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/reset")
async def reset_configuration(
    config_service: ConfigService = Depends(get_config_service)
):
    """Reset configuration to defaults."""
    try:
        success = await config_service.reset_config()
        if success:
            return {"status": "success", "message": "Configuration reset to defaults"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset configuration")
    except Exception as e:
        logger.error(f"Error resetting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert endpoints
@router.get("/alerts")
async def get_alert_history(
    limit: int = 50,
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get alert history."""
    try:
        alerts = await alert_service.get_alert_history(limit)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/stats")
async def get_alert_statistics(
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get alert statistics."""
    try:
        stats = await alert_service.get_alert_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting alert stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/alerts")
async def clear_alert_history(
    alert_service: AlertService = Depends(get_alert_service)
):
    """Clear alert history."""
    try:
        success = await alert_service.clear_alert_history()
        if success:
            return {"status": "success", "message": "Alert history cleared"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear alert history")
    except Exception as e:
        logger.error(f"Error clearing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@router.get("/websocket/status")
async def get_websocket_status(
    websocket_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """Get WebSocket connection status."""
    try:
        stats = websocket_manager.get_connection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health and info endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "arp_detection": arp_service is not None,
            "config": config_service is not None,
            "alerts": alert_service is not None,
            "websocket": websocket_manager is not None
        }
    }

@router.get("/info")
async def get_application_info():
    """Get application information."""
    return {
        "name": "ARP Spoofing Detection Web Application",
        "version": "1.0.0",
        "description": "Real-time ARP spoofing detection and monitoring system",
        "features": [
            "Real-time ARP monitoring",
            "ML-powered detection",
            "Rule-based detection",
            "WebSocket real-time updates",
            "Alert system",
            "Configuration management"
        ]
    }

# Test endpoint
@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {"message": "API is working", "timestamp": datetime.now().isoformat()}

# Test detections endpoint
@router.get("/test-detections")
async def test_detections_endpoint(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Test endpoint to verify detections functionality."""
    try:
        # Test basic service access
        if arp_service is None:
            return {"error": "ARP service not available", "status": "failed"}
        
        # Test getting recent detections
        detections = await arp_service.get_recent_detections(5)
        
        return {
            "status": "success",
            "message": "Detections endpoint working",
            "detections_count": len(detections),
            "service_available": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test detections endpoint failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "service_available": arp_service is not None,
            "timestamp": datetime.now().isoformat()
        }

# Set global service references (called by main.py)
def set_services(
    arp_svc: ARPDetectionService,
    config_svc: ConfigService,
    alert_svc: AlertService,
    ws_manager: WebSocketManager
):
    """Set global service references."""
    global arp_service, config_service, alert_service, websocket_manager
    arp_service = arp_svc
    config_service = config_svc
    alert_service = alert_svc
    websocket_manager = ws_manager 