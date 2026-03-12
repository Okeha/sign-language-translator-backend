"""VideoMAE inference service"""

import base64
import io
import time
import logging
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from src.api.config import config
from src.api.schemas import GlossPrediction

logger = logging.getLogger(__name__)


class VideoMAEService:
    """Singleton service for VideoMAE model inference"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model (only once due to singleton pattern)"""
        if self._initialized:
            return
        
        # logger.info(f"Loading VideoMAE model from {config.MODEL_PATH}")
        
        try:
            # Load model and processor
            self.model = VideoMAEForVideoClassification.from_pretrained(config.MODEL_PATH)
            
            # Try loading processor from checkpoint, fallback to base model if missing
            try:
                self.processor = VideoMAEImageProcessor.from_pretrained(config.MODEL_PATH)
            except Exception as e:
                self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Extract label mapping
            self.id2label = self.model.config.id2label
            
            self._initialized = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def decode_base64_frames(self, frames: List[str]) -> List[np.ndarray]:
        """
        Convert list of base64 JPEG strings to numpy arrays
        
        Args:
            frames: List of base64 encoded JPEG images
            
        Returns:
            List of numpy arrays with shape (H, W, 3) and dtype uint8
        """
        decoded_frames = []
        
        for frame_b64 in frames:
            try:
                # Remove data URI prefix if present
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(frame_b64)
                
                # Open as PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB numpy array
                frame_array = np.array(image.convert('RGB'))
                
                decoded_frames.append(frame_array)
                
            except Exception as e:
                logger.error(f"Failed to decode frame: {str(e)}")
                raise ValueError(f"Invalid frame format: {str(e)}")
        
        return decoded_frames
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Sample 16 frames uniformly and apply VideoMAE preprocessing
        
        Args:
            frames: List of numpy arrays (can be any length >= 1)
            
        Returns:
            Dictionary with 'pixel_values' tensor of shape (1, 16, 3, 224, 224)
        """
        total_frames = len(frames)
        num_frames = config.NUM_FRAMES_TO_SAMPLE  # 16
        
        # Uniformly sample indices
        if total_frames >= num_frames:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            # If fewer frames than needed, repeat frames
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        sampled_frames = [frames[i] for i in indices]
        
        # Ensure uint8 range [0, 255]
        sampled_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in sampled_frames]
        
        # VideoMAEImageProcessor handles resize to 224x224 and normalization
        inputs = self.processor(sampled_frames, return_tensors="pt")
        
        return inputs
    
    def predict(self, frames_b64: List[str]) -> GlossPrediction:
        """
        End-to-end prediction from base64 frames to gloss
        
        Args:
            frames_b64: List of base64 encoded JPEG frames
            
        Returns:
            GlossPrediction object with gloss, confidence, top5, etc.
        """
        start_time = time.time()
        
        try:
            # Step 1: Decode base64 to numpy arrays
            frames = self.decode_base64_frames(frames_b64)
            logger.debug(f"Decoded {len(frames)} frames")
            
            # Step 2: Preprocess (sample 16 frames)
            inputs = self.preprocess_frames(frames)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.debug(f"Preprocessed tensor shape: {inputs['pixel_values'].shape}")
            
            # Step 3: Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-5 predictions
                top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
                top5_probs = top5_probs[0].cpu().numpy()
                top5_indices = top5_indices[0].cpu().numpy()
                
                # Get top-1 prediction
                predicted_idx = torch.argmax(logits, dim=-1).item()
            
            # Step 4: Map indices to glosses
            predicted_gloss = self.id2label[predicted_idx]
            top5_glosses = [
                (self.id2label[idx], float(prob)) 
                for idx, prob in zip(top5_indices, top5_probs)
            ]
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Prediction: {predicted_gloss} ({top5_probs[0]:.2f}) in {latency_ms:.0f}ms")
            
            return GlossPrediction(
                gloss=predicted_gloss,
                confidence=float(top5_probs[0]),
                top5=top5_glosses,
                timestamp=int(time.time() * 1000),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized
