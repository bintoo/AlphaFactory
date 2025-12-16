import logging
import sys
from . import config

def setup_logging():
    """
    Configures the root logger to write to both file and stdout.
    """
    # Create a custom logger
    root_handler = logging.getLogger()
    root_handler.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplication if called multiple times
    if root_handler.hasHandlers():
        root_handler.handlers.clear()

    # Create handlers
    # Force UTF-8 for stdout if possible
    try:
        if sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    c_handler = logging.StreamHandler(sys.stdout)
    # Force UTF-8 for FileHandler
    f_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    root_handler.addHandler(c_handler)
    root_handler.addHandler(f_handler)
    
    logging.info(f"Logging initialized. Writing to {config.LOG_FILE}")
