import logging
from logging.handlers import RotatingFileHandler
from .config import LOG_DIR

LOG_FILE = LOG_DIR / "api.log"

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    return logger
