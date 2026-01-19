import os
import sys
from datetime import datetime
from config.parameters import PATH_LOGS

# Constants for log levels
ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

# ANSI color codes
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

class ColorLogger:
    def __init__(self, log_file_path=None):
        """
        Initializes the color logger
        
        Args:
            log_file_path (str): Path to the log file. If None, no file writing.
        """
        self.log_file_path = log_file_path
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensures the log file directory exists"""
        if self.log_file_path:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                    print(f"{COLOR_BLUE}[INFO] Created log directory: {log_dir}{COLOR_RESET}")
                except Exception as e:
                    print(f"{COLOR_RED}[ERROR] Failed to create log directory {log_dir}: {e}{COLOR_RESET}")
                    self.log_file_path = None  # Disable file logging
    
    def _get_color_code(self, level):
        """Gets the color code based on the level"""
        if level == ERROR:
            return COLOR_RED
        elif level == WARNING:
            return COLOR_YELLOW
        elif level == INFO:
            return COLOR_BLUE
        else:
            return COLOR_RESET
    
    def _format_message(self, message, level, include_date=True):
        """Formats the message with timestamp and level"""
        if include_date:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"[{timestamp}] [{level}] {message}"
        else:
            return f"[{level}] {message}"
    
    def log(self, message, level=INFO):
        """
        Writes a message to the log with color based on level
        
        Args:
            message (str): Message to write to the log
            level (str): Message level (ERROR, WARNING, INFO)
        """
        if level not in [ERROR, WARNING, INFO]:
            level = INFO  # Default level if invalid
        
        formatted_message = self._format_message(message, level, include_date=True)
        color_code = self._get_color_code(level)
        
        # Always show in console with color
        print(f"{color_code}{formatted_message}{COLOR_RESET}")
        
        # Write to file (without color codes)
        if self.log_file_path:
            try:
                # Open file in append mode for each write and close immediately
                with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(formatted_message + '\n')
            except Exception as e:
                # If file writing fails, just show error in console
                error_msg = f"[ERROR] Error writing to log file {self.log_file_path}: {e}"
                print(f"{COLOR_RED}{error_msg}{COLOR_RESET}")

# Global logger instance for better performance
_global_logger = None
_global_log_file = None

def setup_global_logger(log_file_path=None):
    """Setup a global logger instance to be used across files"""
    global _global_logger, _global_log_file
    _global_log_file = log_file_path
    _global_logger = ColorLogger(log_file_path)
    return _global_logger

def get_global_logger():
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        # Create a console-only logger if no global logger set up
        _global_logger = ColorLogger()
    return _global_logger

def quick_log(message, level=INFO, log_file_path=PATH_LOGS):
    """
    Quick function for logging - uses global logger if available,
    otherwise creates a temporary one
    
    Args:
        message (str): Message to log
        level (str): Log level (ERROR, WARNING, INFO)
        log_file_path (str): Optional specific log file path
    """
    if log_file_path:
        # If specific file path provided, create temporary logger
        temp_logger = ColorLogger(log_file_path)
        temp_logger.log(message, level)
    else:
        # Use global logger if no specific file path provided
        get_global_logger().log(message, level)

# Alternative simple version without global logger
def simple_log(message, level=INFO, log_file_path=None):
    """
    Simple logging function that works reliably in all scenarios
    """
    if level not in [ERROR, WARNING, INFO]:
        level = INFO
    
    if log_file_path is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
    else:
        formatted_message = f"[{level}] {message}"
    
    # Console output with colors
    if level == ERROR:
        print(f"{COLOR_RED}{formatted_message}{COLOR_RESET}")
    elif level == WARNING:
        print(f"{COLOR_YELLOW}{formatted_message}{COLOR_RESET}")
    elif level == INFO:
        print(f"{COLOR_BLUE}{formatted_message}{COLOR_RESET}")
    else:
        print(formatted_message)
    
    # File output (if specified)
    if log_file_path:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Write to file
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')
        except Exception as e:
            print(f"{COLOR_RED}[ERROR] Failed to write to log file: {e}{COLOR_RESET}")

# Usage example
if __name__ == "__main__":
    # Test the different logging methods
    quick_log("Testing quick_log with ERROR", ERROR)
    quick_log("Testing quick_log with INFO", INFO)
    
    simple_log("Testing simple_log with WARNING", WARNING, "logs/simple_test.log")
    simple_log("Testing simple_log with ERROR", ERROR, "logs/simple_test.log")
    
    # Test file not found scenario
    quick_log("File not found: /nonexistent/path.txt", ERROR)