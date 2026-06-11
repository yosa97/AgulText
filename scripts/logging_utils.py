import logging
import os
import sys

from colorama import Back, Fore, Style


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = self.COLORS[levelname] + Style.BRIGHT + levelname + Style.RESET_ALL
            record.levelname = levelname_color

        message = super().format(record)

        color = self.COLORS.get(record.levelname, Fore.WHITE)
        message = message.replace("$RESET", Style.RESET_ALL)
        message = message.replace("$BOLD", Style.BRIGHT)
        message = message.replace("$COLOR", color)
        message = message.replace("$BLUE", Fore.BLUE + Style.BRIGHT)

        return message


def get_logger(name: str):
    logger = logging.getLogger(name.split(".")[-1])
    mode: str = os.getenv("ENV", "prod").lower()

    logger.setLevel(logging.DEBUG if mode != "prod" else logging.INFO)
    logger.handlers.clear()

    format_string = (
        "$BLUE%(asctime)s.%(msecs)03d$RESET | "
        "$COLOR$BOLD%(levelname)-8s$RESET | "
        "$BLUE%(name)s$RESET:"
        "$BLUE%(funcName)s$RESET:"
        "$BLUE%(lineno)d$RESET - "
        "$COLOR$BOLD%(message)s$RESET"
    )

    colored_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    logger.debug(f"Logging mode is {logging.getLevelName(logger.getEffectiveLevel())}")
    return logger


def get_training_logger(task_id: str, log_dir: str | None = None) -> logging.Logger:
    """Create a training-run logger that writes both to console (coloured) and
    optionally to a rotating file in *log_dir*.

    Parameters
    ----------
    task_id:  SN56 task identifier; used as the logger name and in the filename.
    log_dir:  Directory for the log file.  Pass None to skip file logging.
    """
    import datetime
    from logging.handlers import RotatingFileHandler

    logger = get_logger(task_id)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{task_id}_{ts}.log")
        fh = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        plain_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(plain_fmt)
        logger.addHandler(fh)
        logger.debug(f"File logging enabled → {log_path}")

    return logger