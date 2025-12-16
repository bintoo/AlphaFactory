import abc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try import external libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class LLMInterface(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class MockLLM(LLMInterface):
    """Mock LLM for testing/offline mode if needed."""
    def generate(self, prompt: str) -> str:
        return "class MyAlgo(QCAlgorithm): pass"

# Placeholder for real LLM clients (OpenAI, DeepSeek, Google)
class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not openai:
            raise ImportError("openai package not installed. Please install it.")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            logger.debug(f"[{self.model}] Sending Request:\n{prompt}")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        # temperature=0.7,
                        timeout=90.0  # Increased timeout
                    )
                    content = response.choices[0].message.content
                    
                    # Log usage if available
                    if response.usage:
                        usage_info = f"Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}"
                        logger.info(f"[{self.model}] Usage: {usage_info}")
                    
                    # logger.debug(f"[{self.model}] Received Response:\n{content}")
                    return content
                    
                except Exception as e:
                    logger.warning(f"API Call Failed (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    else:
                        raise e  # Re-raise on final failure
        except Exception as e:
            logger.exception(f"OpenAI/DeepSeek API Error")
            return f"# Error generating code: {e}"

class DeepSeekLLM(OpenAILLM):
    """DeepSeek is API compatible with OpenAI"""
    def __init__(self, api_key: str, model: str, base_url: str):
        super().__init__(api_key, model, base_url)

class GoogleLLM(LLMInterface):
    def __init__(self, api_key: str, model: str):
        if not genai:
             raise ImportError("google-generativeai package not installed.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str) -> str:
        try:
            logger.debug(f"[Gemini] Sending Request:\n{prompt}")
            response = self.model.generate_content(prompt)
            content = response.text
            # logger.debug(f"[Gemini] Received Response:\n{content}")
            return content
        except Exception as e:
            logger.exception(f"Google API Error")
            return f"# Error generating code: {e}"
