import os

# Configuration for AlphaFactory_Evolution



# Directory for successful exported algorithms
SUCCESSFUL_ALGOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_algos")

# Database for Alpha Storage
DB_PATH = os.path.join(os.path.dirname(__file__), "alpha_db.json")

# Logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "alpha_factory.log")

# LLM Configuration
# Options: "mock", "openai", "deepseek", "google"
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ...

# LLM Configuration
# Options: "mock", "openai", "deepseek", "google"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

# OpenAI Settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

# DeepSeek Settings
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Google Settings
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")

# Data Ranges
TEST_START_DATE = "2020-01-01"
TEST_END_DATE = "2025-01-01"

VALIDATION_START_DATE = "2011-01-01"
VALIDATION_END_DATE = "2021-12-31"

# Performance Goals
TARGET_ANNUAL_RETURN = 0.30  # 30%
TARGET_SHARPE_RATIO = 1.0

# Universe Configuration
UNIVERSE_MODE = "static" 
# TARGET_UNIVERSE = ["NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN","META", "GOOGL", "NFLX", "INTC", "QQQ", "SPY"]
TARGET_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT","IEF", "GLD", "VNQ", "XLE"]
# TARGET_RESOLUTION = "Minute"

# --- STRATEGY SETTINGS ---
# TARGET_FILENAME = "main.py" # Already defined above
TARGET_CLASS_NAME = "MyAgentStrategy"

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
