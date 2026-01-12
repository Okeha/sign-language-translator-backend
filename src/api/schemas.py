"""Pydantic schemas for request/response validation"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional


class FrameBatchRequest(BaseModel):
    """WebSocket incoming message containing video frames"""
    
    session_id: str = Field(..., description="Unique session identifier")
    frames: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=120, 
        description="Base64 JPEG encoded frames"
    )
    timestamp: int = Field(..., description="Client timestamp in milliseconds")
    
    @field_validator('frames')
    @classmethod
    def validate_base64(cls, v: List[str]) -> List[str]:
        """Validate that frames are valid base64 data URIs"""
        for idx, frame in enumerate(v):
            if not frame.startswith('data:image/jpeg;base64,') and not frame.startswith('/9j/'):
                raise ValueError(f'Frame {idx} is not a valid base64 JPEG format')
        return v


class GlossPrediction(BaseModel):
    """Model prediction response"""
    
    gloss: str = Field(..., description="Predicted sign language gloss")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    top5: List[Tuple[str, float]] = Field(..., description="Top 5 predictions with confidences")
    timestamp: int = Field(..., description="Server timestamp in milliseconds")
    latency_ms: float = Field(..., description="Inference time in milliseconds")


class GlossToSentenceRequest(BaseModel):
    """Request to convert glosses to natural sentence"""
    
    session_id: str = Field(..., description="Session identifier")
    glosses: Optional[List[str]] = Field(
        None, 
        description="Glosses to convert (if None, uses all session glosses)"
    )


class SentenceResponse(BaseModel):
    """Generated sentence response"""
    
    sentence: str = Field(..., description="Natural language sentence")
    glosses_used: List[str] = Field(..., description="Glosses used to generate sentence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether VideoMAE model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    active_connections: int = Field(..., description="Number of active WebSocket connections")
    model_path: str = Field(..., description="Path to loaded model")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    timestamp: int = Field(..., description="Error timestamp in milliseconds")


class InterpretGlossesRequest(BaseModel):
    """Request for interpreting gloss sequences to natural language"""
    
    input: List[List[str]] = Field(
        ...,
        description="2D array of glosses. Each inner array represents a video chunk with 5 possible glosses",
        min_length=1
    )
    
    @field_validator('input')
    @classmethod
    def validate_input(cls, v: List[List[str]]) -> List[List[str]]:
        """Validate that each chunk has glosses"""
        for idx, chunk in enumerate(v):
            if len(chunk) == 0:
                raise ValueError(f'Chunk {idx} is empty')
        return v


class ChatRequest(BaseModel):
    """Request for general chat conversation"""
    
    message: str = Field(
        ...,
        description="User's chat message",
        min_length=1,
        max_length=2000
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    
    response: str = Field(..., description="LLM's response to user message")
    timestamp: int = Field(..., description="Response timestamp in milliseconds")


class InterpretGlossesResponse(BaseModel):
    """Response with interpreted sentence"""
    
    sentence: str = Field(..., description="Natural English sentence paraphrased from glosses")
