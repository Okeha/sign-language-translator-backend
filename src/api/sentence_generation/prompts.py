"""Prompt loading and formatting utilities"""

import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.api.config import config

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages loading and formatting prompts from YAML file"""
    
    def __init__(self, prompts_file_path: str = None):
        """
        Initialize prompt manager
        
        Args:
            prompts_file_path: Path to prompts YAML file
        """
        self.prompts_file_path = prompts_file_path or config.PROMPTS_FILE_PATH
        self.prompts_data = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        try:
            with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded prompts from {self.prompts_file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load prompts file: {str(e)}")
            raise
    
    def format_examples(self) -> str:
        """Format few-shot examples into readable text"""
        examples = self.prompts_data.get('examples', [])
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            input_glosses = example['input']
            output_sentence = example['output']
            
            # Format input glosses with explicit array notation
            glosses_text = "Possible glosses (pick ONE per array):\n"
            for chunk_idx, chunk in enumerate(input_glosses, 1):
                glosses_text += f"  Array {chunk_idx}: [{', '.join(chunk)}] → select ONE\n"
            
            # Add output
            formatted_examples.append(
                f"Example {i}:\n{glosses_text}English sentence: {output_sentence}"
            )
        
        return "\n\n".join(formatted_examples)
    
    def format_gloss_input(self, glosses_sequence: List[List[str]]) -> str:
        """
        Format gloss sequence into prompt format
        
        Args:
            glosses_sequence: 2D array of glosses [[chunk1], [chunk2], ...]
            
        Returns:
            Formatted string for prompt
        """
        glosses_text = "Possible glosses (pick ONE per array):\n"
        for chunk_idx, chunk in enumerate(glosses_sequence, 1):
            glosses_text += f"  Array {chunk_idx}: [{', '.join(chunk)}]\n"
        
        return glosses_text
    
    def build_prompt(self, glosses_sequence: List[List[str]]) -> str:
        """
        Build complete prompt for Qwen
        
        Args:
            glosses_sequence: 2D array of glosses
            
        Returns:
            Complete formatted prompt
        """
        system_prompt = self.prompts_data.get('system_prompt', '')
        examples = self.format_examples()
        gloss_input = self.format_gloss_input(glosses_sequence)

        print(gloss_input)
        
        template = self.prompts_data.get('template', '')
        
        prompt = template.format(
            system_prompt=system_prompt,
            # examples=examples,
            gloss_input=gloss_input
        )
        
        return prompt


# Global singleton instance
prompt_manager = PromptManager()
