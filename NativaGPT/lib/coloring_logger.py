import sys
import os
from datetime import datetime
import inspect

class ColoredLogger:
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset color
    }
    
    def __init__(self, name=None):
        if name is None:
            # Get the name of the calling module
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')
        self.name = name
        
        # Check if we're in a terminal that supports colors
        self.use_colors = self._supports_color()
    
    def _supports_color(self):
        """Check if the terminal supports ANSI colors"""
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check for common color-supporting terminals
        term = os.environ.get('TERM', '').lower()
        colorterm = os.environ.get('COLORTERM', '').lower()
        
        return (
            'color' in term or 
            'ansi' in term or 
            'xterm' in term or 
            colorterm in ('truecolor', '24bit', 'yes')
        )
    
    def _log(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            formatted_message = f"{timestamp} - {color}{level:<8}{reset} - {message}"
        else:
            formatted_message = f"{timestamp} - {level:<8} - {message}"
        
        print(formatted_message, file=sys.stdout)
        sys.stdout.flush()
    
    def debug(self, message):
        self._log("DEBUG", str(message))
    
    def info(self, message):
        self._log("INFO", str(message))
    
    def warning(self, message):
        self._log("WARNING", str(message))
    
    def warn(self, message):  # Alias for warning
        self.warning(message)
    
    def error(self, message):
        self._log("ERROR", str(message))
    
    def critical(self, message):
        self._log("CRITICAL", str(message))

# Create a default logger instance that can be imported directly
logger = ColoredLogger()

# Also provide a function to create named loggers if needed
def get_logger(name):
    return ColoredLogger(name)