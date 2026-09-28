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
from src.api.videomae.sampling import (
    decode_uniformly_sampled_frames,
    uniform_sample_indices,
)

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
                # Detect which base model processor to use based on model hidden_size
                # Note: VideoMAE-Huge architecture config exists, but the base model was never
                # officially released on HuggingFace Hub. Use Large processor as fallback since
                # the processor is just for image preprocessing and compatible across sizes.
                hidden_size = self.model.config.hidden_size
                if hidden_size == 1280:  # Huge architecture
                    base_model = "MCG-NJU/videomae-large"  # Use Large processor for Huge (not released)
                    logger.info(f"Huge model detected (hidden_size=1280). Using Large processor (Huge not released on Hub)")
                elif hidden_size == 1024:  # Large architecture
                    base_model = "MCG-NJU/videomae-large"
                else:  # Base or other
                    base_model = "MCG-NJU/videomae-base"
                    
                logger.info(f"Processor not found in checkpoint. Falling back to: {base_model}")
                self.processor = VideoMAEImageProcessor.from_pretrained(base_model)
            
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
    
    def _preprocess_sampled_frames(
        self,
        sampled_frames: List[np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Apply VideoMAE preprocessing to an already sampled frame sequence."""
        sampled_frames = [
            np.clip(frame, 0, 255).astype(np.uint8)
            for frame in sampled_frames
        ]

        return self.processor(sampled_frames, return_tensors="pt")

    def preprocess_frames(self, frames: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Sample frames uniformly and apply VideoMAE preprocessing.
        
        Args:
            frames: List of numpy arrays (can be any length >= 1)
            
        Returns:
            Dictionary with 'pixel_values' tensor of shape
            (1, NUM_FRAMES_TO_SAMPLE, 3, 224, 224)
        """
        indices = uniform_sample_indices(
            len(frames),
            config.NUM_FRAMES_TO_SAMPLE,
        )
        sampled_frames = [frames[index] for index in indices]
        return self._preprocess_sampled_frames(sampled_frames)
    
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
            # Step 1: Select the legacy uniform temporal positions first, then
            # decode only the unique JPEGs referenced by that sample sequence.
            sampled_frames = decode_uniformly_sampled_frames(
                frames_b64,
                config.NUM_FRAMES_TO_SAMPLE,
                self.decode_base64_frames,
            )
            logger.debug(
                "Decoded %s sampled frames from %s source frames",
                len({id(frame) for frame in sampled_frames}),
                len(frames_b64),
            )
            
            # Step 2: Preprocess the already sampled frame sequence.
            inputs = self._preprocess_sampled_frames(sampled_frames)
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
