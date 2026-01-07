"""Qwen sentence generation service for ASL gloss interpretation"""

import logging
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from src.api.config import config
from src.api.sentence_generation.prompts import prompt_manager

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv('HF_TOKEN')
class QwenSentenceService:
    """Singleton service for gloss-to-sentence generation using Qwen2.5-1.5B-Instruct"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Qwen model (only once due to singleton pattern)"""
        if self._initialized:
            return
        
        logger.info(f"Loading Qwen model: {config.LLM_MODEL_NAME}")
        
        try:
            # Check HF token
            if not HF_TOKEN:
                logger.warning("HF_TOKEN not set. Model loading may fail for gated models.")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME,
                token = HF_TOKEN,
                trust_remote_code=True
            )
            
            # Load model with optional quantization
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True,
            }
            
            print("HF_TOKEN:", HF_TOKEN)
            model_kwargs["token"] = HF_TOKEN
            
            if config.LLM_USE_QUANTIZATION:
                logger.info("Loading model with 4-bit quantization...")
                model_kwargs["load_in_4bit"] = True
            else:
                logger.info("Loading model in full precision...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                **model_kwargs,
            )
            
            self.model.eval()
            
            # Get device info
            if hasattr(self.model, 'hf_device_map'):
                self.device = "quantized (4-bit)" if config.LLM_USE_QUANTIZATION else str(self.model.hf_device_map)
            else:
                self.device = next(self.model.parameters()).device
            
            self._initialized = True
            logger.info(f"Qwen model loaded successfully on {self.device}")
            
            # Log memory usage if CUDA
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
        
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {str(e)}")
            raise
    
    def interpret_glosses(self, glosses_sequence: List[List[str]]) -> str:
        """
        Interpret gloss sequence and generate natural language sentence
        
        Args:
            glosses_sequence: 2D array of glosses [[chunk1], [chunk2], ...]
                             Each chunk contains 5 possible gloss interpretations
        
        Returns:
            Natural English sentence (paraphrased)
        """
        try:
            # Build the gloss input text
            gloss_input = prompt_manager.format_gloss_input(glosses_sequence)
            
            # Get system prompt
            system_prompt = prompt_manager.prompts_data.get('system_prompt', '').strip()
            
            # Format as chat messages (Phi-3.5 expects chat format)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these glosses and generate a natural English sentence:\n\n{gloss_input}\n\nEnglish sentence:"}
            ]
            
            logger.debug(f"Generated prompt for {len(glosses_sequence)} chunks")
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking = True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to correct device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.LLM_MAX_LENGTH,
                    temperature=config.LLM_TEMPERATURE,
                    do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # Only decode new tokens
                skip_special_tokens=True
            )
            
            # Clean up output (remove extra whitespace, newlines)
            sentence = generated_text.strip()
            
            # Extract first sentence if multiple sentences generated
            if '.' in sentence:
                sentence = sentence.split('.')[0] + '.'
            elif '!' in sentence:
                sentence = sentence.split('!')[0] + '!'
            elif '?' in sentence:
                sentence = sentence.split('?')[0] + '?'
            
            logger.info(f"Generated sentence: {sentence}")
            
            return sentence
        
        except Exception as e:
            logger.error(f"Failed to interpret glosses: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized
