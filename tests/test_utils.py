import os
import logging
from unittest.mock import patch, MagicMock
from alpha_factory.utils import setup_logging
from alpha_factory import config

def test_setup_logging_creates_files(tmp_path):
    # Override log file path for test
    test_log_file = tmp_path / "test.log"
    
    with patch('alpha_factory.config.LOG_FILE', str(test_log_file)):
        setup_logging()
        
        logging.getLogger().info("Test Log Entry")
        
        # Check if file created and has content
        assert test_log_file.exists()
        content = test_log_file.read_text()
        assert "Test Log Entry" in content
