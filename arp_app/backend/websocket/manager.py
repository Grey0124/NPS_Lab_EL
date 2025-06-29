#!/usr/bin/env python3
"""
WebSocket Manager for Real-time Communication
"""

import asyncio
import json
import logging
import threading
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from queue import Queue
import numpy as np

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

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info = {}  # WebSocket -> connection info
        self.message_queue = Queue()
        self._main_loop = None
        self._queue_processor_task = None
        
    def _get_main_loop(self):
        """Get the main event loop, creating a new one if needed."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def broadcast_sync(self, message: Dict[str, Any]):
        """Thread-safe broadcast method that can be called from any thread."""
        try:
            # Convert numpy types to native Python types
            message = convert_numpy_types(message)
            
            # Put message in queue
            self.message_queue.put(message)
            logger.debug(f"Message queued for broadcast: {message.get('type', 'unknown')}")
            
            # Try to schedule processing in main loop
            try:
                loop = self._get_main_loop()
                if loop.is_running():
                    # Schedule the broadcast in the main loop
                    loop.call_soon_threadsafe(self._process_queued_message)
                else:
                    # If loop is not running, we'll process it later
                    logger.warning("Event loop not running, message will be processed later")
            except Exception as e:
                logger.warning(f"Could not schedule broadcast: {e}")
                
        except Exception as e:
            logger.error(f"Error in broadcast_sync: {e}")
    
    def _process_queued_message(self):
        """Process a single message from the queue."""
        try:
            if not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                # Create a task to broadcast the message
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.broadcast(message))
        except Exception as e:
            logger.error(f"Error processing queued message: {e}")
    
    async def start_queue_processor(self):
        """Start the message queue processor."""
        self._main_loop = asyncio.get_event_loop()
        self._queue_processor_task = asyncio.create_task(self._queue_processor())
    
    async def _queue_processor(self):
        """Background task to process queued messages."""
        while True:
            try:
                # Process any queued messages
                while not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self.broadcast(message)
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)
    
    async def stop_queue_processor(self):
        """Stop the message queue processor."""
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection info
        self.connection_info[websocket] = {
            'connected_at': asyncio.get_event_loop().time(),
            'messages_sent': 0,
            'last_activity': asyncio.get_event_loop().time()
        }
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message(websocket, {
            'type': 'connection_established',
            'message': 'Connected to ARP Detection Service',
            'timestamp': asyncio.get_event_loop().time()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.connection_info:
            del self.connection_info[websocket]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific WebSocket connection."""
        try:
            message_json = json.dumps(message)
            await websocket.send_text(message_json)
            
            # Update connection info
            if websocket in self.connection_info:
                self.connection_info[websocket]['messages_sent'] += 1
                self.connection_info[websocket]['last_activity'] = asyncio.get_event_loop().time()
                
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active WebSocket connections to broadcast to")
            return
        
        # Convert numpy types to native Python types
        message = convert_numpy_types(message)
        
        # Convert message to JSON once
        message_json = json.dumps(message)
        logger.debug(f"Broadcasting message to {len(self.active_connections)} connections: {message.get('type', 'unknown')}")
        
        # Send to all connections
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
                
                # Update connection info
                if connection in self.connection_info:
                    self.connection_info[connection]['messages_sent'] += 1
                    self.connection_info[connection]['last_activity'] = asyncio.get_event_loop().time()
                    
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
        
        if disconnected:
            logger.info(f"Broadcasted to {len(self.active_connections)} connections. {len(disconnected)} disconnected.")
        else:
            logger.debug(f"Successfully broadcasted to {len(self.active_connections)} connections")
    
    async def broadcast_stats(self, stats: Dict[str, Any]):
        """Broadcast statistics update."""
        await self.broadcast({
            'type': 'stats_update',
            'data': stats,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert notification."""
        await self.broadcast({
            'type': 'alert',
            'data': alert,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def broadcast_status(self, status: str, details: Dict[str, Any] = None):
        """Broadcast status update."""
        message = {
            'type': 'status_update',
            'status': status,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        if details:
            message['details'] = details
        
        await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        if not self.connection_info:
            return {
                'total_connections': 0,
                'active_connections': 0,
                'total_messages_sent': 0
            }
        
        total_messages = sum(info['messages_sent'] for info in self.connection_info.values())
        
        return {
            'total_connections': len(self.connection_info),
            'active_connections': len(self.active_connections),
            'total_messages_sent': total_messages,
            'connections': [
                {
                    'connected_at': info['connected_at'],
                    'messages_sent': info['messages_sent'],
                    'last_activity': info['last_activity']
                }
                for info in self.connection_info.values()
            ]
        }
    
    async def cleanup_inactive_connections(self, timeout_seconds: int = 300):
        """Remove inactive connections."""
        current_time = asyncio.get_event_loop().time()
        inactive = []
        
        for connection, info in self.connection_info.items():
            if current_time - info['last_activity'] > timeout_seconds:
                inactive.append(connection)
        
        for connection in inactive:
            self.disconnect(connection)
        
        if inactive:
            logger.info(f"Cleaned up {len(inactive)} inactive connections")
    
    async def shutdown(self):
        """Gracefully shutdown the WebSocket manager."""
        logger.info("Shutting down WebSocket manager...")
        
        # Stop queue processor
        await self.stop_queue_processor()
        
        # Close all active connections
        for connection in self.active_connections[:]:  # Copy list to avoid modification during iteration
            try:
                await connection.close(code=1000, reason="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
        
        # Clear connections
        self.active_connections.clear()
        self.connection_info.clear()
        
        logger.info("WebSocket manager shutdown complete") 