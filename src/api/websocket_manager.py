"""WebSocket connection manager"""

import logging
from typing import Dict
from fastapi import WebSocket

from src.api.schemas import GlossPrediction, ErrorResponse

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        # Maps session_id -> WebSocket object
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """
        Accept and store WebSocket connection
        
        Args:
            session_id: Unique session identifier
            websocket: WebSocket connection object
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"[WebSocket] Session {session_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, session_id: str):
        """
        Remove connection from active list
        
        Args:
            session_id: Session identifier to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"[WebSocket] Session {session_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_prediction(self, session_id: str, prediction: GlossPrediction):
        """
        Send gloss prediction to specific client
        
        Args:
            session_id: Target session identifier
            prediction: GlossPrediction object to send
        """
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(prediction.model_dump())
            except Exception as e:
                logger.error(f"[WebSocket] Failed to send prediction to {session_id}: {str(e)}")
                self.disconnect(session_id)
    
    async def send_error(self, session_id: str, error_message: str):
        """
        Send error message to specific client
        
        Args:
            session_id: Target session identifier
            error_message: Error message to send
        """
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                import time
                error_response = ErrorResponse(
                    error=error_message,
                    timestamp=int(time.time() * 1000)
                )
                await websocket.send_json(error_response.model_dump())
            except Exception as e:
                logger.error(f"[WebSocket] Failed to send error to {session_id}: {str(e)}")
                self.disconnect(session_id)
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Dictionary to send to all clients
        """
        disconnected = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"[WebSocket] Failed to broadcast to {session_id}: {str(e)}")
                disconnected.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected:
            self.disconnect(session_id)
    
    def get_connection_count(self) -> int:
        """
        Get number of active connections
        
        Returns:
            Number of active WebSocket connections
        """
        return len(self.active_connections)
    
    def is_connected(self, session_id: str) -> bool:
        """
        Check if session is connected
        
        Args:
            session_id: Session identifier to check
            
        Returns:
            True if session has active connection
        """
        return session_id in self.active_connections
