"""Qwen sentence generation service for ASL gloss interpretation"""

import logging
import pathlib
import threading
import torch
from typing import List, Dict, Generator, Optional
import os
from openai import OpenAI
import yaml
from src.api.config import config
from src.api.sentence_generation.prompts import prompt_manager

logger = logging.getLogger(__name__)

# load prompts.yml from this package's folder
_prompts_path = pathlib.Path(__file__).parent / "prompts.yml"

with open(_prompts_path, 'r', encoding='utf-8') as f:
    model_signing_prompt = yaml.safe_load(f)["model_signing_prompt"]


MODEL_NAME = "minimax/minimax-m2.5:free"

CHAT_SYSTEM_PROMPT = (
    "You are Signrr. A helpful AI assistant that provides clear, concise responses "
    "and bridges the digital gap between sign language users and AI technology. "
    "The user may ask questions about how to use the system. Encourage them to start "
    "the camera, and click the start streaming button twice. However, reply in a "
    "concise manner. Minimize giving long explanations unless necessary and answer questions asked or statements "
    "made by the user. Only encourage them to start streaming signs if they ask about "
    "it or mention it. If the user does not ask about the sign language system, respond "
    "to their query directly without mentioning sign language or the system."
    "If asked about what model you are say something like I am Signrr, developed by Anthony Okeh."
    "Please ensure that if asked about what the time is, you always give a time even if it is fake."
    "Most times the user communicates with you via the camera stream, so you don't have to mention it everytime."
    "Try to always give your response in a nice structure (markdown is encouraged please)"
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
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": role, "content": content})

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            logger.info(f"Chat memory cleared for session: {session_id}")

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get_message_count(self, session_id: str) -> int:
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
        self.chat_memory = ChatMemory()
        self.model_signing_prompt = model_signing_prompt
        self.MODEL_NAME = MODEL_NAME
        self.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY",
            ""
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.OPENROUTER_API_KEY,
        )

        self._initialized = True

    def interpret_glosses(self, glosses_sequence: List[List[str]]) -> str:
        """
        Interpret gloss sequence and generate natural language sentence.

        Args:
            glosses_sequence: 2D array of glosses [[chunk1], [chunk2], ...]
                             Each chunk contains 5 possible gloss interpretations

        Returns:
            Natural English sentence (paraphrased)
        """
        try:
            gloss_input = prompt_manager.format_gloss_input(glosses_sequence)
            system_prompt = prompt_manager.prompts_data.get("system_prompt", "").strip()

            expected_format = (
                "Only respond with final sentence. The sentence should be concise and clear. "
                "Remember the constraints STEP 1 — SELECT: Pick exactly ONE word from each array. "
                "- Higher confidence words are more likely correct. "
                "- BUT when scores are low (under 4%), the model is very uncertain. "
                "In this case, weigh cross-array coherence MORE heavily — pick the combination "
                "of words that forms a real sentence, even if those words have lower scores. "
                "STEP 2 — TRANSLATE: Take your selected words IN ORDER and form a natural English sentence. "
                "## IMPORTANT: DO NOT use more than one word from the same array!"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the arrays:\n\n{gloss_input}\n\nRemember: {expected_format}."},
            ]

            logger.debug(f"Generated prompt for {len(glosses_sequence)} chunks")

            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
            )

            return response.choices[0].message.content

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
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

            if session_id:
                history = self.chat_memory.get_history(session_id)
                messages.extend(history)

            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise

    def chat_stream(
        self, user_message: str, session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        try:
            print(f"Received chat message: {user_message} for session: {session_id}")

            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

            if session_id and hasattr(self, 'chat_memory'):
                history = self.chat_memory.get_history(session_id)
                messages.extend(history)

            messages.append({"role": "user", "content": user_message})

            full_response = ""

            with self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
                stream=True,
            ) as stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        full_response += content
                        yield content

            if session_id and hasattr(self, 'chat_memory'):
                self.chat_memory.add_message(session_id, "user", user_message)
                self.chat_memory.add_message(session_id, "assistant", full_response)

        except Exception as e:
            logger.error(f"Error in chat stream: {str(e)}")
            yield f"Error: {str(e)}"

    def clear_chat_memory(self, session_id: str) -> None:
        """Clear chat memory for a session."""
        self.chat_memory.clear_session(session_id)

    def convert_chat_to_gloss(self, chat_text: str) -> str:
        """
        Convert chat text to gloss format.

        Args:
            chat_text: Input chat text

        Returns:
            str: Gloss formatted text
        """
        try:
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
                "'YELLOW', 'YESTERDAY', 'YOU']. \n ## NEVER ADD WORDS NOT IN THE VOCABULARY LIST BUT TRY TO USE THE VOCABULARY WORDS AS MUCH AS POSSIBLE. Always give a response. Ensure what you generate is actually cohesive please and represents the sentence to a large extent. \n - Use the closest synonyms to the sentence found in the vocabulary words if you cannot match exactly."
            )

            messages = [
                {"role": "system", "content": self.model_signing_prompt},
                {"role": "user", "content": chat_text},
            ]

            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._initialized