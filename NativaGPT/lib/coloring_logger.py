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

    # === MÉTODO PRINCIPAL ADAPTADO ===
    def _log(self, level, message, file=sys.stdout):
        """
        Método de logging principal, agora com suporte ao parâmetro 'file'.
        O 'file' padrão é sys.stdout, mas pode ser sys.stderr ou outro stream.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            formatted_message = f"{timestamp} - {color}{level:<8}{reset} - {message}"
        else:
            formatted_message = f"{timestamp} - {level:<8} - {message}"

        # Usa o stream 'file' fornecido para a saída
        print(formatted_message, file=file)
        file.flush()

    # === MÉTODOS PÚBLICOS ADAPTADOS ===
    def debug(self, message, file=sys.stdout):
        self._log("DEBUG", str(message), file)

    def info(self, message, file=sys.stdout):
        self._log("INFO", str(message), file)

    def warning(self, message, file=sys.stdout):
        self._log("WARNING", str(message), file)

    def warn(self, message, file=sys.stdout):  # Alias for warning
        self.warning(message, file)

    def error(self, message, file=sys.stderr): # ERRO por padrão vai para stderr
        self._log("ERROR", str(message), file)

    def critical(self, message, file=sys.stderr): # CRITICAL por padrão vai para stderr
        self._log("CRITICAL", str(message), file)

# Create a default logger instance that can be imported directly
logger = ColoredLogger()

# Also provide a function to create named loggers if needed
def get_logger(name):
    return ColoredLogger(name)