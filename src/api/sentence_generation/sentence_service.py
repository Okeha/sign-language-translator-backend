"""Qwen sentence generation service for ASL gloss interpretation"""

import logging
import pathlib
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

import yaml

from src.api.config import config
from src.api.sentence_generation.prompts import prompt_manager

logger = logging.getLogger(__name__)

# HF_TOKEN = os.getenv('HF_TOKEN')
# print("\n\n HF_TOKEN:", HF_TOKEN)

# load prompts.yml from this package's folder
_prompts_path = pathlib.Path(__file__).parent / "prompts.yml"

with open(_prompts_path, 'r', encoding='utf-8') as f:
    model_signing_prompt = yaml.safe_load(f)["model_signing_prompt"]

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
        
        logger.info(f"Loading Model: {config.LLM_MODEL_NAME}")

       
        self.model_signing_prompt = model_signing_prompt        

            # print(model_signing_prompt)
        
        try:
            # Check HF token
            if not config.HF_TOKEN:
                logger.warning("HF_TOKEN not set. Model loading may fail for gated models.")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME,
                token = config.HF_TOKEN,
                trust_remote_code=True
            )
            
            # Load model with optional quantization
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": False,
            }
            
            if config.HF_TOKEN:
                model_kwargs["token"] = config.HF_TOKEN
            
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
                {"role": "user", "content": f"Here are the arrays:\n\n{gloss_input}"}
            ]
            
            logger.debug(f"Generated prompt for {len(glosses_sequence)} chunks")
            
            # # Apply chat template
            # prompt = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     enable_thinking = True
            # )
            
            # # Tokenize
            # inputs = self.tokenizer(
            #     prompt,
            #     return_tensors="pt",
            #     truncation=True,
            #     max_length=2048
            # )
            
            # # Move to correct device
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # # Generate
            # with torch.no_grad():
            #     outputs = self.model.generate(
            #         **inputs,
            #         max_new_tokens=config.LLM_MAX_LENGTH,
            #         temperature=config.LLM_TEMPERATURE,
            #         do_sample=True if config.LLM_TEMPERATURE > 0 else False,
            #         top_p=0.9,
            #         pad_token_id=self.tokenizer.eos_token_id
            #     )
            
            # # Decode
            # generated_text = self.tokenizer.decode(
            #     outputs[0][inputs['input_ids'].shape[1]:],  # Only decode new tokens
            #     skip_special_tokens=True
            # )


            # Alternative: Use built-in chat generation

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking = False, # TODO: Enable reasoning if needed
                tokenize = True,
                return_dict = True,
                return_tensors = "pt",
            ).to(device)
            
            # Clean up output (remove extra whitespace, newlines)
            outputs = self.model.generate(
                **inputs,
                # max_new_tokens=config.LLM_MAX_LENGTH,
                max_new_tokens = 30000,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_text=self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Extract only the text after thinking tags (if present)
            if '<think>' in generated_text and '</think>' in generated_text:
                # Get everything after the closing think tag
                generated_text = generated_text.split('</think>')[-1].strip()

            sentence = generated_text.strip()
            print("\n\n Generated Text:", generated_text)
            
            sentence = sentence.replace('"', '').replace("'", '')
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
    
    def chat(self, user_message: str) -> str:
        """Generate a response to user's chat message without reasoning
        
        Args:
            user_message: User's chat message
            
        Returns:
            str: LLM's response
        """
        try:
            device = self.model.device
            
            # Simple chat messages without complex system prompt
            messages = [
                {"role": "system", "content": "You are a Signrr. A helpful AI assistant that provides clear, concise responses and bridges the digital gap between sign language users and AI technology. The user may ask questions about how to use the system. Encourage them to start the camera, and click the start streaming button twice. However, reply in a concise manner. Avoid long explanations and answer questions asked or statements made by the user. Only encourage them to start streaming signs if they ask about it or mention it. If the user does not ask about the sign language system, respond to their query directly without mentioning sign language or the system."},
                {"role": "user", "content": user_message}
            ]
            
            # Apply chat template without reasoning enabled
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,  # No reasoning for simple chat
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Generate response with temperature control
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,  # Shorter responses for chat
                temperature=config.LLM_TEMPERATURE,
                do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Chat response generated: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise
    
    def convert_chat_to_gloss(self, chat_text:str) -> str:
        """Convert chat text to gloss format by replacing spaces with underscores
        
        Args:
            chat_text: Input chat text
            
        Returns:
            str: Gloss formatted text
        """
        try:
            device = self.model.device
            
            # Simple chat messages without complex system prompt
            messages = [
                {"role": "system", "content": self.model_signing_prompt},
                {"role": "user", "content": chat_text}
            ]
            
            # Apply chat template without reasoning enabled
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,  # No reasoning for simple gloss conversion
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Generate response with temperature control
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,  # Shorter responses for chat
                temperature=0.7,
                do_sample=True if 0.5 > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Chat response generated: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise


    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized
