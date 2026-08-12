"""Lightweight ANSI-colored console logger.

Provides ``ColoredLogger``, a minimal logger that timestamps messages and
colors them by severity level when the output stream is a color-capable
terminal, plus a ready-to-use module-level ``logger`` instance and a
``get_logger(name)`` factory for creating additional named instances.
"""

import sys
import os
from datetime import datetime
import inspect

class ColoredLogger:
    """Minimal logger that timestamps and color-codes messages by level.

    Messages are printed directly with ``print`` (no ``logging`` module
    involved). When stdout is a terminal that appears to support ANSI
    colors, each severity level is rendered in its own color; otherwise
    plain text is used.

    Attributes:
        name (str): Logical name of the logger, typically the calling
            module's ``__name__`` when not explicitly provided.
        use_colors (bool): Whether ANSI color codes are emitted, based on
            the detected terminal capabilities.
        COLORS (dict): Class-level mapping of level name (and ``'RESET'``)
            to ANSI escape codes.
    """

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
        """Initializes the logger.

        Args:
            name: Logical name for the logger. If ``None``, the name is
                inferred from the ``__name__`` of the calling module's
                globals (falling back to ``'unknown'``).
        """
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
        """Logs a message at DEBUG level.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stdout``.
        """
        self._log("DEBUG", str(message), file)

    def info(self, message, file=sys.stdout):
        """Logs a message at INFO level.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stdout``.
        """
        self._log("INFO", str(message), file)

    def warning(self, message, file=sys.stdout):
        """Logs a message at WARNING level.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stdout``.
        """
        self._log("WARNING", str(message), file)

    def warn(self, message, file=sys.stdout):  # Alias for warning
        """Alias for :meth:`warning`.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stdout``.
        """
        self.warning(message, file)

    def error(self, message, file=sys.stderr): # ERRO por padrão vai para stderr
        """Logs a message at ERROR level.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stderr``.
        """
        self._log("ERROR", str(message), file)

    def critical(self, message, file=sys.stderr): # CRITICAL por padrão vai para stderr
        """Logs a message at CRITICAL level.

        Args:
            message: Message to log (converted to ``str``).
            file: Output stream to write to. Defaults to ``sys.stderr``.
        """
        self._log("CRITICAL", str(message), file)

# Create a default logger instance that can be imported directly
logger = ColoredLogger()

# Also provide a function to create named loggers if needed
def get_logger(name):
    """Creates a new named :class:`ColoredLogger` instance.

    Args:
        name: Logical name to associate with the returned logger.

    Returns:
        ColoredLogger: A new logger instance using the given name.
    """
    return ColoredLogger(name)