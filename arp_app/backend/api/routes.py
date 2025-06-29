#!/usr/bin/env python3
"""
API Routes for ARP Detection Web Application
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

# Import services (these will be injected)
from services.arp_detector_service import ARPDetectionService
from services.config_service import ConfigService
from services.alert_service import AlertService
from services.registry_service import RegistryService
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
registry_service: Optional[RegistryService] = None
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

def get_registry_service() -> RegistryService:
    if registry_service is None:
        raise HTTPException(status_code=503, detail="Registry service not available")
    return registry_service

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
        
        # Add registry statistics if detector is available
        if arp_service.detector:
            registry_stats = arp_service.detector.get_registry_stats()
            stats['registry'] = registry_stats
        
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics/registry")
async def get_registry_statistics(
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Get registry-specific statistics."""
    try:
        if not arp_service.detector:
            return {
                "auto_registry_addition": False,
                "registry_additions": 0,
                "total_registry_entries": 0,
                "arp_history_size": 0,
                "status": "detector_not_available"
            }
        
        registry_stats = arp_service.detector.get_registry_stats()
        return registry_stats
    except Exception as e:
        logger.error(f"Error getting registry statistics: {e}")
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
        updated_sections = []
        
        if request.detection:
            success &= await config_service.update_detection_config(request.detection)
            updated_sections.append("detection")
        
        if request.alerts:
            success &= await config_service.update_alert_config(request.alerts)
            updated_sections.append("alerts")
        
        if success:
            # If monitoring is active, restart with new config
            if arp_service.is_monitoring and arp_service.current_interface:
                await arp_service.stop_monitoring()
                await arp_service.start_monitoring(arp_service.current_interface)
            
            return {
                "status": "success", 
                "message": f"Configuration updated: {', '.join(updated_sections)}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
            
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/reset")
async def reset_configuration(
    config_service: ConfigService = Depends(get_config_service),
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Reset configuration to defaults."""
    try:
        success = await config_service.reset_config()
        if success:
            # If monitoring is active, restart with new config
            if arp_service.is_monitoring and arp_service.current_interface:
                await arp_service.stop_monitoring()
                await arp_service.start_monitoring(arp_service.current_interface)
            
            return {"status": "success", "message": "Configuration reset to defaults"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset configuration")
    except Exception as e:
        logger.error(f"Error resetting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/export")
async def export_configuration(
    config_service: ConfigService = Depends(get_config_service)
):
    """Export configuration as JSON."""
    try:
        config = await config_service.get_full_config()
        return {
            "status": "success",
            "config": config,
            "exported_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Error exporting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/import")
async def import_configuration(
    request: dict,
    config_service: ConfigService = Depends(get_config_service),
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Import configuration from JSON."""
    try:
        if not request.get("config"):
            raise HTTPException(status_code=400, detail="No configuration data provided")
        
        success = await config_service.import_config(request["config"])
        if success:
            # If monitoring is active, restart with new config
            if arp_service.is_monitoring and arp_service.current_interface:
                await arp_service.stop_monitoring()
                await arp_service.start_monitoring(arp_service.current_interface)
            
            return {"status": "success", "message": "Configuration imported successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to import configuration")
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/validate")
async def validate_configuration(
    config_service: ConfigService = Depends(get_config_service)
):
    """Validate current configuration."""
    try:
        validation_result = await config_service.validate_config()
        return {
            "status": "success",
            "valid": validation_result["valid"],
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", [])
        }
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/backup")
async def backup_configuration(
    config_service: ConfigService = Depends(get_config_service)
):
    """Create a backup of current configuration."""
    try:
        backup_path = await config_service.create_backup()
        return {
            "status": "success",
            "message": "Configuration backup created",
            "backup_path": backup_path
        }
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/backups")
async def list_backups(
    config_service: ConfigService = Depends(get_config_service)
):
    """List available configuration backups."""
    try:
        backups = await config_service.list_backups()
        return {
            "status": "success",
            "backups": backups
        }
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config/restore/{backup_name}")
async def restore_configuration(
    backup_name: str,
    config_service: ConfigService = Depends(get_config_service),
    arp_service: ARPDetectionService = Depends(get_arp_service)
):
    """Restore configuration from backup."""
    try:
        success = await config_service.restore_backup(backup_name)
        if success:
            # If monitoring is active, restart with new config
            if arp_service.is_monitoring and arp_service.current_interface:
                await arp_service.stop_monitoring()
                await arp_service.start_monitoring(arp_service.current_interface)
            
            return {"status": "success", "message": f"Configuration restored from {backup_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to restore configuration")
    except Exception as e:
        logger.error(f"Error restoring config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config/schema")
async def get_configuration_schema():
    """Get configuration schema for validation."""
    return {
        "detection": {
            "type": "object",
            "properties": {
                "registry_path": {"type": "string"},
                "model_path": {"type": "string"},
                "log_file": {"type": "string"},
                "detection_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "max_queue_size": {"type": "integer", "minimum": 100, "maximum": 100000},
                "alert_cooldown": {"type": "integer", "minimum": 0, "maximum": 3600},
                "auto_registry_addition": {"type": "boolean"}
            },
            "required": ["registry_path", "model_path", "detection_threshold"]
        },
        "alerts": {
            "type": "object",
            "properties": {
                "email_enabled": {"type": "boolean"},
                "email_recipients": {"type": "array", "items": {"type": "string"}},
                "webhook_enabled": {"type": "boolean"},
                "webhook_url": {"type": "string", "format": "uri"},
                "notification_cooldown": {"type": "integer", "minimum": 0, "maximum": 3600}
            }
        },
        "web": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "debug": {"type": "boolean"},
                "cors_origins": {"type": "array", "items": {"type": "string"}}
            }
        },
        "database": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "echo": {"type": "boolean"}
            }
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "format": {"type": "string"},
                "file": {"type": "string"}
            }
        }
    }

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

# Test detections endpoint - COMMENTED OUT: Automatic registry addition is now working properly
# @router.get("/test-detections")
# async def test_detections_endpoint(
#     arp_service: ARPDetectionService = Depends(get_arp_service)
# ):
#     """Test endpoint to verify detections functionality."""
#     try:
#         # Test basic service access
#         if arp_service is None:
#             return {"error": "ARP service not available", "status": "failed"}
#         
#         # Test getting recent detections
#         detections = await arp_service.get_recent_detections(5)
#         
#         return {
#             "status": "success",
#             "message": "Detections endpoint working",
#             "detections_count": len(detections),
#             "service_available": True,
#             "timestamp": datetime.now().isoformat()
#         }
#         
#     except Exception as e:
#         logger.error(f"Test detections endpoint failed: {e}")
#         return {
#             "status": "failed",
#             "error": str(e),
#             "service_available": arp_service is not None,
#             "timestamp": datetime.now().isoformat()
#         }

# Registry management endpoints
@router.get("/registry")
async def get_registry_entries(
    registry_service: RegistryService = Depends(get_registry_service)
):
    """Get all registry entries."""
    try:
        entries = await registry_service.get_entries()
        return {
            "status": "success",
            "entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        logger.error(f"Error getting registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/registry/direct")
async def get_registry_entries_direct():
    """Get all registry entries directly from file."""
    try:
        registry_path = Path("data/registry.yml")
        entries = {}
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        # Parse simple "key: value" format
                        parts = line.split(':', 1)  # Split on first colon only
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                entries[key] = value
        
        return {
            "status": "success",
            "entries": entries,
            "count": len(entries),
            "source": "direct_file_read"
        }
    except Exception as e:
        logger.error(f"Error getting registry directly: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/registry/export")
async def export_registry():
    """Export registry as JSON."""
    try:
        registry_path = Path("data/registry.yml")
        entries = {}
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        # Parse simple "key: value" format
                        parts = line.split(':', 1)  # Split on first colon only
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                entries[key] = value
        
        return {
            "status": "success",
            "entries": entries,
            "exported_at": datetime.now().isoformat(),
            "count": len(entries),
            "source": "direct_file_read"
        }
    except Exception as e:
        logger.error(f"Error exporting registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/registry/import")
async def import_registry(request: dict):
    """Import registry from JSON."""
    try:
        if not request.get("entries"):
            raise HTTPException(status_code=400, detail="No registry entries provided")
        
        registry_path = Path("data/registry.yml")
        entries = request["entries"]
        
        # Write entries directly to file
        with open(registry_path, 'w') as f:
            for ip, mac in entries.items():
                f.write(f"{ip}: {mac}\n")
        
        return {
            "status": "success", 
            "message": f"Registry imported successfully. Added {len(entries)} entries.",
            "source": "direct_file_write"
        }
    except Exception as e:
        logger.error(f"Error importing registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/registry/add")
async def add_registry_entry(request: dict):
    """Add a single registry entry."""
    try:
        ip = request.get("ip")
        mac = request.get("mac")
        
        if not ip or not mac:
            raise HTTPException(status_code=400, detail="IP and MAC addresses are required")
        
        registry_path = Path("data/registry.yml")
        entries = {}
        
        # Read existing entries
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                entries[key] = value
        
        # Check if entry already exists
        if ip in entries and entries[ip] == mac:
            return {
                "status": "info",
                "message": f"Entry {ip} -> {mac} already exists"
            }
        
        # Add new entry
        entries[ip] = mac
        
        # Write back to file
        with open(registry_path, 'w') as f:
            for entry_ip, entry_mac in entries.items():
                f.write(f"{entry_ip}: {entry_mac}\n")
        
        return {
            "status": "success",
            "message": f"Added {ip} -> {mac} to registry"
        }
    except Exception as e:
        logger.error(f"Error adding registry entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/registry/remove/{ip}")
async def remove_registry_entry(ip: str):
    """Remove a registry entry."""
    try:
        registry_path = Path("data/registry.yml")
        entries = {}
        
        # Read existing entries
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line and not line.startswith('#'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                entries[key] = value
        
        # Check if entry exists
        if ip not in entries:
            return {
                "status": "info",
                "message": f"No entry found for {ip}"
            }
        
        # Remove entry
        del entries[ip]
        
        # Write back to file
        with open(registry_path, 'w') as f:
            for entry_ip, entry_mac in entries.items():
                f.write(f"{entry_ip}: {entry_mac}\n")
        
        return {
            "status": "success",
            "message": f"Removed {ip} from registry"
        }
    except Exception as e:
        logger.error(f"Error removing registry entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/registry/reset")
async def reset_registry():
    """Reset registry (clear all entries)."""
    try:
        registry_path = Path("data/registry.yml")
        
        # Clear the file
        with open(registry_path, 'w') as f:
            pass  # Write empty file
        
        return {
            "status": "success",
            "message": "Registry reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting registry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Set global service references (called by main.py)
def set_services(
    arp_svc: ARPDetectionService,
    config_svc: ConfigService,
    alert_svc: AlertService,
    registry_svc: RegistryService,
    ws_manager: WebSocketManager
):
    """Set global service references."""
    global arp_service, config_service, alert_service, registry_service, websocket_manager
    arp_service = arp_svc
    config_service = config_svc
    alert_service = alert_svc
    registry_service = registry_svc
    websocket_manager = ws_manager 