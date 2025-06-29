#!/usr/bin/env python3
"""
FastAPI Backend for ARP Spoofing Detection Web Application
"""

import asyncio
import json
import logging
import os
import sys
import signal
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from services.arp_detector_service import ARPDetectionService
from services.config_service import ConfigService
from services.alert_service import AlertService
from services.registry_service import RegistryService
from models.database import init_db
from api.routes import router as api_router, set_services
from websocket.manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ARP Spoofing Detection API",
    description="Real-time ARP spoofing detection and monitoring system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with more robust configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:5173",  # Alternative localhost
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:8080",  # Alternative port
        "http://127.0.0.1:8080",  # Alternative port
        "*"  # Allow all origins for development (fallback)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global services
arp_service: Optional[ARPDetectionService] = None
config_service: Optional[ConfigService] = None
alert_service: Optional[AlertService] = None
registry_service: Optional[RegistryService] = None
websocket_manager: Optional[WebSocketManager] = None

# Shutdown event
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global arp_service, config_service, alert_service, registry_service, websocket_manager
    
    logger.info("Starting ARP Spoofing Detection Web Application...")
    
    # Initialize database
    await init_db()
    
    # Initialize services
    config_service = ConfigService()
    alert_service = AlertService()
    registry_service = RegistryService()
    websocket_manager = WebSocketManager()
    
    # Start WebSocket queue processor
    await websocket_manager.start_queue_processor()
    
    # Initialize ARP detection service
    arp_service = ARPDetectionService(
        config_service=config_service,
        alert_service=alert_service,
        websocket_manager=websocket_manager
    )
    
    # Set service references in API routes
    set_services(arp_service, config_service, alert_service, registry_service, websocket_manager)
    
    logger.info("All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event_handler():
    """Cleanup on shutdown."""
    global arp_service, websocket_manager
    
    logger.info("Shutting down ARP Spoofing Detection Web Application...")
    
    # Stop ARP monitoring
    if arp_service:
        await arp_service.stop_monitoring()
    
    # Stop WebSocket manager
    if websocket_manager:
        await websocket_manager.shutdown()
    
    logger.info("Shutdown complete")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Simple health check endpoint (no service dependencies)
@app.get("/ping")
async def ping():
    """Simple ping endpoint to test basic connectivity."""
    return {
        "status": "ok",
        "message": "pong",
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ARP Spoofing Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler that ensures CORS headers are sent."""
    logger.error(f"Unhandled exception: {exc}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Return JSON response with CORS headers
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler with CORS headers."""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down...")
    finally:
        logger.info("Application stopped") 