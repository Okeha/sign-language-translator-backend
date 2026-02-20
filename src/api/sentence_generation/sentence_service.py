"""Qwen sentence generation service for ASL gloss interpretation"""

import logging
import pathlib
import threading
import torch
from typing import List, Dict, Generator, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import os

import yaml

from src.api.config import config
from src.api.sentence_generation.prompts import prompt_manager

logger = logging.getLogger(__name__)

# load prompts.yml from this package's folder
_prompts_path = pathlib.Path(__file__).parent / "prompts.yml"

with open(_prompts_path, 'r', encoding='utf-8') as f:
    model_signing_prompt = yaml.safe_load(f)["model_signing_prompt"]


CHAT_SYSTEM_PROMPT = (
    "You are Signrr. A helpful AI assistant that provides clear, concise responses "
    "and bridges the digital gap between sign language users and AI technology. "
    "The user may ask questions about how to use the system. Encourage them to start "
    "the camera, and click the start streaming button twice. However, reply in a "
    "concise manner. Minimize giving long explanations unless necessary and answer questions asked or statements "
    "made by the user. Only encourage them to start streaming signs if they ask about "
    "it or mention it. If the user does not ask about the sign language system, respond "
    "to their query directly without mentioning sign language or the system."
    "If asked about what model you are say; I am Signrr, developed by Anthony Okeh"
)


class ChatMemory:
    """
    Per-session chat history manager.
    
    Stores conversation history in memory. Clears when session ends
    or server restarts. No persistence.
    """

    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to session history."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({
                "role": role,
                "content": content,
            })

    def clear_session(self, session_id: str) -> None:
        """Clear history for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
            logger.info(f"Chat memory cleared for session: {session_id}")

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return session_id in self._sessions

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def get_message_count(self, session_id: str) -> int:
        """Get number of messages in a session."""
        with self._lock:
            return len(self._sessions.get(session_id, []))


class QwenSentenceService:
    """Singleton service for gloss-to-sentence generation using Qwen"""

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
        self.chat_memory = ChatMemory()

        try:
            # Check HF token
            if not config.HF_TOKEN:
                logger.warning("HF_TOKEN not set. Model loading may fail for gated models.")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME,
                token=config.HF_TOKEN,
                trust_remote_code=True,
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
            if hasattr(self.model, "hf_device_map"):
                self.device = (
                    "quantized (4-bit)"
                    if config.LLM_USE_QUANTIZATION
                    else str(self.model.hf_device_map)
                )
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
            system_prompt = prompt_manager.prompts_data.get("system_prompt", "").strip()

            # Format as chat messages
            expected_format = "Only respond with final sentence. The sentence should be concise and clear."
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Here are the arrays:\n\n{gloss_input} \n\n Remember: {expected_format}. ",
                },
            ]

            logger.debug(f"Generated prompt for {len(glosses_sequence)} chunks")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30000,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            # Extract only the text after thinking tags (if present)
            if "<think>" in generated_text and "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1].strip()

            sentence = generated_text.strip()
            print("\n\n Generated Text:", generated_text)

            sentence = sentence.replace('"', "").replace("'", "")
            # Extract first sentence if multiple sentences generated
            if "." in sentence:
                sentence = sentence.split(".")[0] + "."
            elif "!" in sentence:
                sentence = sentence.split("!")[0] + "!"
            elif "?" in sentence:
                sentence = sentence.split("?")[0] + "?"

            logger.info(f"Generated sentence: {sentence}")

            return sentence

        except Exception as e:
            logger.error(f"Failed to interpret glosses: {str(e)}")
            raise

    def chat(self, user_message: str, session_id: Optional[str] = None) -> str:
        """
        Generate a response to user's chat message with session memory.

        Args:
            user_message: User's chat message
            session_id: Session ID for conversation memory. If None, no memory.

        Returns:
            str: LLM's response
        """
        try:
            device = self.model.device

            # Build messages with history
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

            # Add conversation history if session exists
            if session_id:
                history = self.chat_memory.get_history(session_id)
                messages.extend(history)

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            response = response.strip()

            # Save to memory if session provided
            if session_id:
                self.chat_memory.add_message(session_id, "user", user_message)
                self.chat_memory.add_message(session_id, "assistant", response)
                msg_count = self.chat_memory.get_message_count(session_id)
                logger.info(f"[{session_id}] Chat memory: {msg_count} messages")

            logger.info(f"Chat response generated: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise

    def chat_stream(
        self, user_message: str, session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream a response to user's chat message token by token with session memory.

        Args:
            user_message: User's chat message
            session_id: Session ID for conversation memory. If None, no memory.

        Yields:
            str: Individual tokens as they are generated
        """
        try:
            device = self.model.device

            # Build messages with history
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

            # Add conversation history if session exists
            if session_id:
                history = self.chat_memory.get_history(session_id)
                messages.extend(history)

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Generation kwargs
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=500,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True if config.LLM_TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            # Run generation in a separate thread (model.generate blocks)
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
            )
            thread.start()

            # Collect full response for memory while yielding tokens
            full_response = ""
            for token in streamer:
                print(f"[DEBUG] Streamed token: {token}")
                full_response += token
                yield token

            thread.join()

            # Save to memory after full response is generated
            if session_id:
                self.chat_memory.add_message(session_id, "user", user_message)
                self.chat_memory.add_message(
                    session_id, "assistant", full_response.strip()
                )
                msg_count = self.chat_memory.get_message_count(session_id)
                logger.info(f"[{session_id}] Chat memory: {msg_count} messages")

            logger.info(f"Streamed response: {full_response.strip()[:100]}...")

        except Exception as e:
            logger.error(f"Error in chat stream: {str(e)}")
            raise

    def clear_chat_memory(self, session_id: str) -> None:
        """Clear chat memory for a session."""
        self.chat_memory.clear_session(session_id)

    def convert_chat_to_gloss(self, chat_text: str) -> str:
        """
        Convert chat text to gloss format

        Args:
            chat_text: Input chat text

        Returns:
            str: Gloss formatted text
        """
        try:
            device = self.model.device

            chat_text = (
                chat_text
                + "Remember the  - VOCABULARY LIST: ['ABOUT', 'AFRAID', 'AFTER', 'AFTERNOON', "
                "'AGAIN', 'ALL', 'ALWAYS', 'AND', 'ANGRY', 'ANSWER', 'ARRIVE', 'ASK', "
                "'BABY', 'BAD', 'BALL', 'BANK', 'BATHROOM', 'BECAUSE', 'BED', 'BEFORE', "
                "'BIG', 'BLACK', 'BLUE', 'BODY', 'BOOK', 'BORED', 'BOSS', 'BOTTLE', "
                "'BOX', 'BOY', 'BRAVE', 'BRING', 'BROTHER', 'BROWN', 'BUILDING', 'BUS', "
                "'BUT', 'CALL', 'CALM', 'CAMERA', 'CAN', 'CAR', 'CARD', 'CARE', "
                "'CARRY', 'CHAIR', 'CHANGE', 'CHILD', 'CHOICE', 'CHURCH', 'CITY', "
                "'CLEAN', 'CLOSE', 'CLOTHES', 'COLD', 'COLOR', 'COME', 'COMPUTER', "
                "'CONFUSED', 'CONTINUE', 'COUNTRY', 'CUP', 'CUSTOMER', 'DANGEROUS', "
                "'DAY', 'DIFFERENT', 'DIRTY', 'DOCTOR', 'DOOR', 'DRINK', 'DROP', "
                "'EARLY', 'EASY', 'EAT', 'EIGHT', 'EMPTY', 'ENOUGH', 'ENTER', 'EVENT', "
                "'EXCITED', 'FAMILY', 'FARM', 'FAST', 'FATHER', 'FEEL', 'FEW', 'FIND', "
                "'FIRE', 'FIRST', 'FIVE', 'FOLLOW', 'FOOD', 'FOR', 'FOUR', 'FRIEND', "
                "'FROM', 'FULL', 'GIRL', 'GIVE', 'GO', 'GOAL', 'GOOD', 'GREEN', "
                "'GROUP', 'HAPPY', 'HARD', 'HATE', 'HAVE', 'HELP', 'HIGH', 'HIT', "
                "'HOME', 'HOPE', 'HOSPITAL', 'HOT', 'HOUR', 'HOUSE', 'HOW', 'HURT', "
                "'I', 'IDEA', 'IF', 'IMPORTANT', 'JUMP', 'KEY', 'KITCHEN', 'KNOW', "
                "'LANGUAGE', 'LAST', 'LATE', 'LEAVE', 'LESS', 'LIBRARY', 'LIFT', "
                "'LIGHT', 'LIKE', 'LONELY', 'LONG', 'LOUD', 'LOVE', 'MAKE', 'MAN', "
                "'MANY', 'MAYBE', 'ME', 'MEDICINE', 'MIND', 'MINE', 'MINUTE', 'MONEY', "
                "'MONTH', 'MORE', 'MORNING', 'MOTHER', 'MOVE', 'MUST', 'NAME', 'NEED', "
                "'NERVOUS', 'NEVER', 'NEW', 'NEXT', 'NIGHT', 'NINE', 'NONE', 'NOW', "
                "'NUMBER', 'NURSE', 'OLD', 'ONE', 'OPEN', 'OR', 'ORANGE', 'PAPER', "
                "'PEOPLE', 'PERSON', 'PHONE', 'PINK', 'PLACE', 'PLAN', 'PLAY', "
                "'POLICE', 'POOR', 'PROBLEM', 'PROUD', 'PULL', 'PURPLE', 'PUSH', "
                "'PUT', 'QUESTION', 'QUIET', 'REASON', 'RED', 'RELAX', 'RESTAURANT', "
                "'RICH', 'RIGHT', 'ROOM', 'RUN', 'SAD', 'SAFE', 'SAME', 'SCHOOL', "
                "'SECOND', 'SEE', 'SEVEN', 'SHE', 'SHOES', 'SHORT', 'SHOULD', 'SICK', "
                "'SIGN', 'SISTER', 'SIX', 'SLEEP', 'SLOW', 'SMALL', 'SOME', "
                "'SOMETIMES', 'SOON', 'START', 'STOP', 'STORE', 'STORY', 'STREET', "
                "'STRONG', 'STUDENT', 'TABLE', 'TAKE', 'TEACHER', 'TEAM', 'TELL', "
                "'TEN', 'THAT', 'THEM', 'THEY', 'THING', 'THINK', 'THIS', 'THREE', "
                "'TIME', 'TIRED', 'TODAY', 'TOMORROW', 'TOUCH', 'TRAIN', 'TRUE', "
                "'TRY', 'TURN', 'TV', 'TWO', 'USE', 'WAIT', 'WALK', 'WANT', 'WATCH', "
                "'WATER', 'WAY', 'WE', 'WEAK', 'WEEK', 'WHAT', 'WHEN', 'WHERE', "
                "'WHICH', 'WHITE', 'WHO', 'WHY', 'WILL', 'WINDOW', 'WITH', 'WITHOUT', "
                "'WOMAN', 'WORK', 'WORKER', 'WORLD', 'WORRY', 'WRONG', 'YEAR', "
                "'YELLOW', 'YESTERDAY', 'YOU'] "
            )
            messages = [
                {"role": "system", "content": self.model_signing_prompt},
                {"role": "user", "content": chat_text},
            ]

            # Apply chat template without reasoning enabled
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            # Generate response with temperature control
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True if 0.5 > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )

            logger.info(f"Chat response generated: {response[:100]}...")
            return response.strip()

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized























# """Qwen sentence generation service for ASL gloss interpretation"""

# import logging
# import pathlib
# import torch
# from typing import List
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# import yaml

# from src.api.config import config
# from src.api.sentence_generation.prompts import prompt_manager

# logger = logging.getLogger(__name__)

# # HF_TOKEN = os.getenv('HF_TOKEN')
# # print("\n\n HF_TOKEN:", HF_TOKEN)

# # load prompts.yml from this package's folder
# _prompts_path = pathlib.Path(__file__).parent / "prompts.yml"

# with open(_prompts_path, 'r', encoding='utf-8') as f:
#     model_signing_prompt = yaml.safe_load(f)["model_signing_prompt"]

# class QwenSentenceService:
#     """Singleton service for gloss-to-sentence generation using Qwen2.5-1.5B-Instruct"""
    
#     _instance = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance
    
#     def __init__(self):
#         """Initialize Qwen model (only once due to singleton pattern)"""
#         if self._initialized:
#             return
        
#         logger.info(f"Loading Model: {config.LLM_MODEL_NAME}")

       
#         self.model_signing_prompt = model_signing_prompt        

#             # print(model_signing_prompt)
        
#         try:
#             # Check HF token
#             if not config.HF_TOKEN:
#                 logger.warning("HF_TOKEN not set. Model loading may fail for gated models.")
            
#             # Load tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 config.LLM_MODEL_NAME,
#                 token = config.HF_TOKEN,
#                 trust_remote_code=True
#             )
            
#             # Load model with optional quantization
#             model_kwargs = {
#                 "device_map": "auto",
#                 "torch_dtype": "auto",
#                 "trust_remote_code": False,
#             }
            
#             if config.HF_TOKEN:
#                 model_kwargs["token"] = config.HF_TOKEN
            
#             if config.LLM_USE_QUANTIZATION:
#                 logger.info("Loading model with 4-bit quantization...")
#                 model_kwargs["load_in_4bit"] = True
#             else:
#                 logger.info("Loading model in full precision...")
            
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 config.LLM_MODEL_NAME,
#                 **model_kwargs,
#             )
            
#             self.model.eval()
            
#             # Get device info
#             if hasattr(self.model, 'hf_device_map'):
#                 self.device = "quantized (4-bit)" if config.LLM_USE_QUANTIZATION else str(self.model.hf_device_map)
#             else:
#                 self.device = next(self.model.parameters()).device
            
#             self._initialized = True
#             logger.info(f"Qwen model loaded successfully on {self.device}")
            
#             # Log memory usage if CUDA
#             if torch.cuda.is_available():
#                 memory_allocated = torch.cuda.memory_allocated() / 1024**3
#                 logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
        
#         except Exception as e:
#             logger.error(f"Failed to load Qwen model: {str(e)}")
#             raise
    
#     def interpret_glosses(self, glosses_sequence: List[List[str]]) -> str:
#         """
#         Interpret gloss sequence and generate natural language sentence
        
#         Args:
#             glosses_sequence: 2D array of glosses [[chunk1], [chunk2], ...]
#                              Each chunk contains 5 possible gloss interpretations
        
#         Returns:
#             Natural English sentence (paraphrased)
#         """
#         try:
#             # Build the gloss input text
#             gloss_input = prompt_manager.format_gloss_input(glosses_sequence)
            
#             # Get system prompt
#             system_prompt = prompt_manager.prompts_data.get('system_prompt', '').strip()
            
#             # Format as chat messages (Phi-3.5 expects chat format)
#             expected_format = "Only respond with final sentence. The sentence should be concise and clear."
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Here are the arrays:\n\n{gloss_input} \n\n Remember: {expected_format}. "}
#             ]
            
#             logger.debug(f"Generated prompt for {len(glosses_sequence)} chunks")


#             # Alternative: Use built-in chat generation

#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             inputs = self.tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 enable_thinking = False, # TODO: Enable reasoning if needed
#                 tokenize = True,
#                 return_dict = True,
#                 return_tensors = "pt",
#             ).to(device)
            
#             # Clean up output (remove extra whitespace, newlines)
#             outputs = self.model.generate(
#                 **inputs,
#                 # max_new_tokens=config.LLM_MAX_LENGTH,
#                 max_new_tokens = 30000,
#                 temperature=config.LLM_TEMPERATURE,
#                 do_sample=True if config.LLM_TEMPERATURE > 0 else False,
#                 top_p=0.9,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#             generated_text=self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
#             # Extract only the text after thinking tags (if present)
#             if '<think>' in generated_text and '</think>' in generated_text:
#                 # Get everything after the closing think tag
#                 generated_text = generated_text.split('</think>')[-1].strip()

#             sentence = generated_text.strip()
#             print("\n\n Generated Text:", generated_text)
            
#             sentence = sentence.replace('"', '').replace("'", '')
#             # Extract first sentence if multiple sentences generated
#             if '.' in sentence:
#                 sentence = sentence.split('.')[0] + '.'
#             elif '!' in sentence:
#                 sentence = sentence.split('!')[0] + '!'
#             elif '?' in sentence:
#                 sentence = sentence.split('?')[0] + '?'
            
#             logger.info(f"Generated sentence: {sentence}")
            
#             return sentence
        
#         except Exception as e:
#             logger.error(f"Failed to interpret glosses: {str(e)}")
#             raise
    
#     def chat(self, user_message: str) -> str:
#         """Generate a response to user's chat message without reasoning
        
#         Args:
#             user_message: User's chat message
            
#         Returns:
#             str: LLM's response
#         """
#         try:
#             device = self.model.device
            
#             # Simple chat messages without complex system prompt
#             messages = [
#                 {"role": "system", "content": "You are a Signrr. A helpful AI assistant that provides clear, concise responses and bridges the digital gap between sign language users and AI technology. The user may ask questions about how to use the system. Encourage them to start the camera, and click the start streaming button twice. However, reply in a concise manner. Avoid long explanations and answer questions asked or statements made by the user. Only encourage them to start streaming signs if they ask about it or mention it. If the user does not ask about the sign language system, respond to their query directly without mentioning sign language or the system."},
#                 {"role": "user", "content": user_message}
#             ]
            
#             # Apply chat template without reasoning enabled
#             inputs = self.tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 enable_thinking=False,  # No reasoning for simple chat
#                 tokenize=True,
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
            
#             # Generate response with temperature control
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=500,  # Shorter responses for chat
#                 temperature=config.LLM_TEMPERATURE,
#                 do_sample=True if config.LLM_TEMPERATURE > 0 else False,
#                 top_p=0.9,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
            
#             # Decode response
#             response = self.tokenizer.decode(
#                 outputs[0][inputs["input_ids"].shape[-1]:],
#                 skip_special_tokens=True
#             )
            
#             logger.info(f"Chat response generated: {response[:100]}...")
#             return response.strip()
            
#         except Exception as e:
#             logger.error(f"Error in chat generation: {str(e)}")
#             raise
    
#     def convert_chat_to_gloss(self, chat_text:str) -> str:
#         """Convert chat text to gloss format by replacing spaces with underscores
        
#         Args:
#             chat_text: Input chat text
            
#         Returns:
#             str: Gloss formatted text
#         """
#         try:
#             device = self.model.device
            
#             # Simple chat messages without complex system prompt

#             chat_text = chat_text + "Remember the  - VOCABULARY LIST: ['ABOUT', 'AFRAID', 'AFTER', 'AFTERNOON', 'AGAIN', 'ALL', 'ALWAYS', 'AND', 'ANGRY', 'ANSWER', 'ARRIVE', 'ASK', 'BABY', 'BAD', 'BALL', 'BANK', 'BATHROOM', 'BECAUSE', 'BED', 'BEFORE', 'BIG', 'BLACK', 'BLUE', 'BODY', 'BOOK', 'BORED', 'BOSS', 'BOTTLE', 'BOX', 'BOY', 'BRAVE', 'BRING', 'BROTHER', 'BROWN', 'BUILDING', 'BUS', 'BUT', 'CALL', 'CALM', 'CAMERA', 'CAN', 'CAR', 'CARD', 'CARE', 'CARRY', 'CHAIR', 'CHANGE', 'CHILD', 'CHOICE', 'CHURCH', 'CITY', 'CLEAN', 'CLOSE', 'CLOTHES', 'COLD', 'COLOR', 'COME', 'COMPUTER', 'CONFUSED', 'CONTINUE', 'COUNTRY', 'CUP', 'CUSTOMER', 'DANGEROUS', 'DAY', 'DIFFERENT', 'DIRTY', 'DOCTOR', 'DOOR', 'DRINK', 'DROP', 'EARLY', 'EASY', 'EAT', 'EIGHT', 'EMPTY', 'ENOUGH', 'ENTER', 'EVENT', 'EXCITED', 'FAMILY', 'FARM', 'FAST', 'FATHER', 'FEEL', 'FEW', 'FIND', 'FIRE', 'FIRST', 'FIVE', 'FOLLOW', 'FOOD', 'FOR', 'FOUR', 'FRIEND', 'FROM', 'FULL', 'GIRL', 'GIVE', 'GO', 'GOAL', 'GOOD', 'GREEN', 'GROUP', 'HAPPY', 'HARD', 'HATE', 'HAVE', 'HELP', 'HIGH', 'HIT', 'HOME', 'HOPE', 'HOSPITAL', 'HOT', 'HOUR', 'HOUSE', 'HOW', 'HURT', 'I', 'IDEA', 'IF', 'IMPORTANT', 'JUMP', 'KEY', 'KITCHEN', 'KNOW', 'LANGUAGE', 'LAST', 'LATE', 'LEAVE', 'LESS', 'LIBRARY', 'LIFT', 'LIGHT', 'LIKE', 'LONELY', 'LONG', 'LOUD', 'LOVE', 'MAKE', 'MAN', 'MANY', 'MAYBE', 'ME', 'MEDICINE', 'MIND', 'MINE', 'MINUTE', 'MONEY', 'MONTH', 'MORE', 'MORNING', 'MOTHER', 'MOVE', 'MUST', 'NAME', 'NEED', 'NERVOUS', 'NEVER', 'NEW', 'NEXT', 'NIGHT', 'NINE', 'NONE', 'NOW', 'NUMBER', 'NURSE', 'OLD', 'ONE', 'OPEN', 'OR', 'ORANGE', 'PAPER', 'PEOPLE', 'PERSON', 'PHONE', 'PINK', 'PLACE', 'PLAN', 'PLAY', 'POLICE', 'POOR', 'PROBLEM', 'PROUD', 'PULL', 'PURPLE', 'PUSH', 'PUT', 'QUESTION', 'QUIET', 'REASON', 'RED', 'RELAX', 'RESTAURANT', 'RICH', 'RIGHT', 'ROOM', 'RUN', 'SAD', 'SAFE', 'SAME', 'SCHOOL', 'SECOND', 'SEE', 'SEVEN', 'SHE', 'SHOES', 'SHORT', 'SHOULD', 'SICK', 'SIGN', 'SISTER', 'SIX', 'SLEEP', 'SLOW', 'SMALL', 'SOME', 'SOMETIMES', 'SOON', 'START', 'STOP', 'STORE', 'STORY', 'STREET', 'STRONG', 'STUDENT', 'TABLE', 'TAKE', 'TEACHER', 'TEAM', 'TELL', 'TEN', 'THAT', 'THEM', 'THEY', 'THING', 'THINK', 'THIS', 'THREE', 'TIME', 'TIRED', 'TODAY', 'TOMORROW', 'TOUCH', 'TRAIN', 'TRUE', 'TRY', 'TURN', 'TV', 'TWO', 'USE', 'WAIT', 'WALK', 'WANT', 'WATCH', 'WATER', 'WAY', 'WE', 'WEAK', 'WEEK', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHITE', 'WHO', 'WHY', 'WILL', 'WINDOW', 'WITH', 'WITHOUT', 'WOMAN', 'WORK', 'WORKER', 'WORLD', 'WORRY', 'WRONG', 'YEAR', 'YELLOW', 'YESTERDAY', 'YOU'] "
#             messages = [
#                 {"role": "system", "content": self.model_signing_prompt},
#                 {"role": "user", "content": chat_text}
#             ]
            
#             # Apply chat template without reasoning enabled
#             inputs = self.tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 enable_thinking=False,  # No reasoning for simple gloss conversion
#                 tokenize=True,
#                 return_dict=True,
#                 return_tensors="pt",
#             ).to(device)
            
#             # Generate response with temperature control
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=500,  # Shorter responses for chat
#                 temperature=0.7,
#                 do_sample=True if 0.5 > 0 else False,
#                 top_p=0.9,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
            
#             # Decode response
#             response = self.tokenizer.decode(
#                 outputs[0][inputs["input_ids"].shape[-1]:],
#                 skip_special_tokens=True
#             )
            
#             logger.info(f"Chat response generated: {response[:100]}...")
#             return response.strip()
            
#         except Exception as e:
#             logger.error(f"Error in chat generation: {str(e)}")
#             raise


#     def is_loaded(self) -> bool:
#         """Check if model is loaded and ready"""
#         return self._initialized
