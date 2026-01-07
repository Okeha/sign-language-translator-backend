"""Session storage for gloss buffers"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.api.config import config

logger = logging.getLogger(__name__)


@dataclass
class GlossEntry:
    """Single gloss entry in session buffer"""
    gloss: str
    confidence: float
    timestamp: int  # milliseconds since epoch


@dataclass
class SessionData:
    """Session data container"""
    session_id: str
    glosses: List[GlossEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class SessionStore:
    """In-memory storage for session gloss buffers"""
    
    def __init__(self):
        # Maps session_id -> SessionData
        self.sessions: Dict[str, SessionData] = {}
    
    def create_session(self, session_id: str):
        """
        Initialize new session with empty gloss list
        
        Args:
            session_id: Unique session identifier
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(session_id=session_id)
            logger.info(f"[SessionStore] Created session {session_id}")
    
    def add_gloss(self, session_id: str, gloss: str, confidence: float):
        """
        Add gloss to session buffer with optional consecutive deduplication
        
        Args:
            session_id: Session identifier
            gloss: Predicted gloss text
            confidence: Prediction confidence
        """
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        # Deduplicate consecutive glosses if enabled
        if config.DEDUPLICATE_CONSECUTIVE:
            if session.glosses and session.glosses[-1].gloss == gloss:
                logger.debug(f"[SessionStore] Skipping duplicate consecutive gloss '{gloss}' for {session_id}")
                return
        
        # Add new gloss entry
        import time
        entry = GlossEntry(
            gloss=gloss,
            confidence=confidence,
            timestamp=int(time.time() * 1000)
        )
        session.glosses.append(entry)
        session.last_activity = datetime.now()
        
        # Limit buffer size (keep last N glosses)
        if len(session.glosses) > config.MAX_GLOSSES_PER_SESSION:
            session.glosses = session.glosses[-config.MAX_GLOSSES_PER_SESSION:]
            logger.debug(f"[SessionStore] Trimmed gloss buffer for {session_id}")
        
        logger.info(f"[SessionStore] Added gloss '{gloss}' to {session_id} (total: {len(session.glosses)})")
    
    def get_glosses(self, session_id: str, last_n: Optional[int] = None) -> List[GlossEntry]:
        """
        Retrieve glosses for session
        
        Args:
            session_id: Session identifier
            last_n: Optional limit to return only last N glosses
            
        Returns:
            List of GlossEntry objects
        """
        if session_id not in self.sessions:
            logger.warning(f"[SessionStore] Session {session_id} not found")
            return []
        
        glosses = self.sessions[session_id].glosses
        
        if last_n:
            return glosses[-last_n:]
        
        return glosses
    
    def get_glosses_as_strings(self, session_id: str, last_n: Optional[int] = None) -> List[str]:
        """
        Retrieve glosses as simple string list
        
        Args:
            session_id: Session identifier
            last_n: Optional limit to return only last N glosses
            
        Returns:
            List of gloss strings
        """
        entries = self.get_glosses(session_id, last_n)
        return [entry.gloss for entry in entries]
    
    def clear_session(self, session_id: str):
        """
        Delete session data
        
        Args:
            session_id: Session identifier to remove
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"[SessionStore] Cleared session {session_id}")
    
    def clear_glosses(self, session_id: str):
        """
        Clear glosses but keep session
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].glosses = []
            self.sessions[session_id].last_activity = datetime.now()
            logger.info(f"[SessionStore] Cleared glosses for session {session_id}")
    
    def cleanup_inactive(self, max_age_hours: Optional[int] = None):
        """
        Remove sessions inactive for more than max_age_hours
        
        Args:
            max_age_hours: Maximum age in hours (defaults to config.SESSION_TIMEOUT_HOURS)
        """
        if max_age_hours is None:
            max_age_hours = config.SESSION_TIMEOUT_HOURS
        
        now = datetime.now()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            age_hours = (now - session.last_activity).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        if to_remove:
            logger.info(f"[SessionStore] Cleaned up {len(to_remove)} inactive sessions")
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions
