"""Sign Language Detection API - Main FastAPI Application"""

import logging
import time
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.api.config import config
from src.api.schemas import (
    FrameBatchRequest,
    GlossPrediction,
    GlossToSentenceRequest,
    SentenceResponse,
    HealthResponse,
    ErrorResponse,
    InterpretGlossesRequest,
    InterpretGlossesResponse
)
from src.api.videomae.model_service import VideoMAEService
from src.api.websocket_manager import ConnectionManager
from src.api.session_store import SessionStore
from src.api.sentence_generation.sentence_service import QwenSentenceService

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sign Language Detection API",
    description="Real-time sign language gloss detection using VideoMAE",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized on startup)
model_service: Optional[VideoMAEService] = None
connection_manager: Optional[ConnectionManager] = None
session_store: Optional[SessionStore] = None
sentence_service: Optional[QwenSentenceService] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on server startup
    - Load VideoMAE model
    - Load Qwen sentence generation model
    - Initialize connection manager
    - Initialize session store
    """
    global model_service, connection_manager, session_store, sentence_service
    
    logger.info("=" * 60)
    logger.info("Starting Sign Language Detection API")
    logger.info("=" * 60)
    
    # Initialize connection manager and session store
    logger.info("Initializing managers...")
    connection_manager = ConnectionManager()
    session_store = SessionStore()
    
    # Load VideoMAE model (this takes 5-10 seconds)
    logger.info(f"Loading VideoMAE model from: {config.MODEL_PATH}")
    try:
        model_service = VideoMAEService()
        logger.info(f"✓ VideoMAE loaded successfully on {model_service.device}")
    except Exception as e:
        logger.error(f"✗ Failed to load VideoMAE: {str(e)}")
        raise
    
    # Load Qwen sentence generation model (this takes 3-5 seconds)
    logger.info(f"Loading Qwen model: {config.LLM_MODEL_NAME}")
    try:
        sentence_service = QwenSentenceService()
        logger.info(f"✓ Qwen loaded successfully on {sentence_service.device}")
    except Exception as e:
        logger.error(f"✗ Failed to load Qwen: {str(e)}")
        logger.warning("Sentence generation will not be available")
        sentence_service = None
    
    logger.info("=" * 60)
    logger.info("API ready to accept connections")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    logger.info(f"VideoMAE Device: {model_service.device}")
    if sentence_service:
        logger.info(f"Qwen Device: {sentence_service.device}")
    logger.info(f"CORS Origins: {config.CORS_ORIGINS}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Shutting down Sign Language Detection API...")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sign Language Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "websocket_stream": "/ws/stream/{session_id}",
            "interpret_glosses": "POST /interpret-glosses",
            "sentence_generation": "POST /glosses/to-sentence",
            "video_upload": "POST /predict/video",
            "session_glosses": "GET /session/{session_id}/glosses",
            "health": "GET /health",
            "ready": "GET /ready",
            "docs": "/docs"
        }
    }


@app.post("/interpret-glosses", response_model=InterpretGlossesResponse)
async def interpret_glosses(request: InterpretGlossesRequest):
    """
    Interpret ASL gloss sequences and generate natural English sentence
    
    Takes a 2D array of glosses where each inner array represents a video chunk
    with 5 possible gloss interpretations. The LLM analyzes all possibilities
    and generates a natural, paraphrased sentence.
    
    Request body:
    {
        "input": [
            ["I", "WE", "CLAP", "SEE", "NOTE"],
            ["WANT", "POOR", "CAT", "DEAL"],
            ["FOOD", "BABY", "CARD", "WE", "YOU"]
        ]
    }
    
    Response:
    {
        "sentence": "I am hungry."
    }
    """
    if not sentence_service or not sentence_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Sentence generation service not available. Qwen model failed to load."
        )
    
    try:
        logger.info(f"Interpreting {len(request.input)} gloss chunks")
        
        # Generate sentence using Qwen
        sentence = sentence_service.interpret_glosses(request.input)
        
        logger.info(f"Generated sentence: {sentence}")
        
        return InterpretGlossesResponse(sentence=sentence)
        
    except Exception as e:
        logger.error(f"Failed to interpret glosses: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentence generation failed: {str(e)}"
        )


@app.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time video frame streaming
    
    Client sends: {"session_id": "...", "frames": [...], "timestamp": 123456}
    Server responds: {"gloss": "book", "confidence": 0.85, "top5": [...], "timestamp": 123456, "latency_ms": 245}
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
    """
    # Accept connection
    await connection_manager.connect(session_id, websocket)
    session_store.create_session(session_id)
    
    logger.info(f"[{session_id}] WebSocket connected")
    
    try:
        # Main message loop
        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()
            
            # Validate request
            try:
                request = FrameBatchRequest(**data)
            except Exception as e:
                error_msg = f"Invalid request format: {str(e)}"
                logger.error(f"[{session_id}] {error_msg}")
                await connection_manager.send_error(session_id, error_msg)
                continue
            
            # Log received frames
            logger.debug(f"[{session_id}] Received {len(request.frames)} frames")
            
            # Run inference
            try:
                prediction = model_service.predict(request.frames)
                
                # Apply confidence threshold
                if prediction.confidence < config.CONFIDENCE_THRESHOLD:
                    logger.debug(
                        f"[{session_id}] Low confidence prediction ({prediction.confidence:.2f}) - "
                        f"skipping (threshold: {config.CONFIDENCE_THRESHOLD})"
                    )
                    continue
                
                # Store gloss in session buffer
                session_store.add_gloss(
                    session_id=session_id,
                    gloss=prediction.gloss,
                    confidence=prediction.confidence
                )
                
                # Send prediction to client
                await connection_manager.send_prediction(session_id, prediction)
                
                logger.info(
                    f"[{session_id}] Predicted: {prediction.gloss} "
                    f"({prediction.confidence:.2f}) in {prediction.latency_ms:.0f}ms"
                )
                
            except Exception as e:
                error_msg = f"Inference failed: {str(e)}"
                logger.error(f"[{session_id}] {error_msg}")
                await connection_manager.send_error(session_id, error_msg)
    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
        connection_manager.disconnect(session_id)
        session_store.clear_session(session_id)
    
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error: {str(e)}")
        connection_manager.disconnect(session_id)
        session_store.clear_session(session_id)


@app.post("/glosses/to-sentence", response_model=SentenceResponse)
async def glosses_to_sentence(request: GlossToSentenceRequest):
    """
    Convert accumulated glosses to natural language sentence
    
    Request body:
    {
        "session_id": "abc123",
        "glosses": ["I", "WANT", "BOOK"]  // optional, uses all session glosses if not provided
    }
    
    Response:
    {
        "sentence": "I want a book.",
        "glosses_used": ["I", "WANT", "BOOK"],
        "confidence": 0.85
    }
    
    TODO: Integrate LLM (Phi-3, Llama, or GPT-4o-mini) for actual sentence generation
    """
    # Check if session exists
    if not session_store.session_exists(request.session_id):
        raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
    
    # Get glosses to use
    if request.glosses and len(request.glosses) > 0:
        glosses_to_use = request.glosses
    else:
        # Use all glosses from session
        glosses_to_use = session_store.get_glosses_as_strings(request.session_id)
    
    if not glosses_to_use:
        raise HTTPException(
            status_code=400, 
            detail="No glosses provided and no glosses found in session"
        )
    
    # PLACEHOLDER: Simple sentence generation
    # TODO: Replace with LLM integration (Phi-3, Llama-3.2, or GPT-4o-mini)
    sentence = " ".join(glosses_to_use).lower().capitalize()
    if not sentence.endswith('.'):
        sentence += '.'
    
    logger.info(f"[{request.session_id}] Generated sentence: {sentence}")
    
    return SentenceResponse(
        sentence=sentence,
        glosses_used=glosses_to_use,
        confidence=0.75  # Placeholder confidence
    )


@app.post("/predict/video", response_model=GlossPrediction)
async def predict_video(file: UploadFile = File(...)):
    """
    Upload video file for prediction (fallback for non-WebSocket clients)
    
    Accepts: .mp4, .avi, .mov, .mkv, .webm
    
    Response: GlossPrediction with gloss and confidence
    """
    import cv2
    import tempfile
    import os
    
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Extract frames from video
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        logger.info(f"Extracted {len(frames)} frames from uploaded video")
        
        # Preprocess and predict (model_service.predict expects base64, so we'll call preprocess directly)
        inputs = model_service.preprocess_frames(frames)
        inputs = {k: v.to(model_service.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model_service.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
            top5_probs = top5_probs[0].cpu().numpy()
            top5_indices = top5_indices[0].cpu().numpy()
            
            predicted_idx = torch.argmax(logits, dim=-1).item()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Map to glosses
        predicted_gloss = model_service.id2label[predicted_idx]
        top5_glosses = [
            (model_service.id2label[idx], float(prob))
            for idx, prob in zip(top5_indices, top5_probs)
        ]
        
        logger.info(f"Video prediction: {predicted_gloss} ({top5_probs[0]:.2f})")
        
        return GlossPrediction(
            gloss=predicted_gloss,
            confidence=float(top5_probs[0]),
            top5=top5_glosses,
            timestamp=int(time.time() * 1000),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Video prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/session/{session_id}/glosses")
async def get_session_glosses(session_id: str, last_n: Optional[int] = None):
    """
    Retrieve stored glosses for a session (debugging endpoint)
    
    Query params:
        last_n: Optional limit to return only last N glosses
    
    Response:
    {
        "session_id": "abc123",
        "gloss_count": 15,
        "glosses": [
            {"gloss": "book", "confidence": 0.85, "timestamp": 123456789},
            ...
        ]
    }
    """
    if not session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    glosses = session_store.get_glosses(session_id, last_n)
    
    return {
        "session_id": session_id,
        "gloss_count": len(glosses),
        "glosses": [
            {
                "gloss": g.gloss,
                "confidence": g.confidence,
                "timestamp": g.timestamp
            }
            for g in glosses
        ]
    }


@app.delete("/session/{session_id}/glosses")
async def clear_session_glosses(session_id: str):
    """
    Clear glosses for a session (keeps session alive)
    
    Useful for resetting detection without disconnecting WebSocket
    """
    if not session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session_store.clear_glosses(session_id)
    
    logger.info(f"[{session_id}] Glosses cleared via REST API")
    
    return {"message": f"Glosses cleared for session {session_id}"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    
    Returns service status, model load status, GPU availability, etc.
    """
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "unhealthy",
        model_loaded=model_service.is_loaded(),
        gpu_available=torch.cuda.is_available(),
        active_connections=connection_manager.get_connection_count(),
        model_path=config.MODEL_PATH
    )


@app.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe
    
    Returns 200 if model is loaded and ready, 503 otherwise
    """
    if model_service and model_service.is_loaded():
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded yet")


@app.get("/stats")
async def get_stats():
    """
    Get current API statistics
    
    Returns active connections, sessions, and system info
    """
    return {
        "active_connections": connection_manager.get_connection_count(),
        "active_sessions": session_store.get_session_count(),
        "model_loaded": model_service.is_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "device": str(model_service.device) if model_service else "unknown",
        "config": {
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "max_glosses_per_session": config.MAX_GLOSSES_PER_SESSION,
            "deduplicate_consecutive": config.DEDUPLICATE_CONSECUTIVE
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )
