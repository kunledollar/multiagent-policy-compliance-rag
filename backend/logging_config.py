import logging
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
import uuid

# Ensure log directory exists
LOG_DIR = "/var/log/rag_system"
os.makedirs(LOG_DIR, exist_ok=True)


# JSON Formatter
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "agent": getattr(record, "agent", None),
            "pipeline_step": getattr(record, "pipeline_step", None),
        }
        return json.dumps(log_record)


def get_rotating_handler(log_filename):
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=5_000_000,  # 5 MB per log file
        backupCount=5
    )
    handler.setFormatter(JSONFormatter())
    return handler


# MAIN LOGGER
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)
api_logger.addHandler(get_rotating_handler("api.log"))


# ERROR LOGGER
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_logger.addHandler(get_rotating_handler("errors.log"))


# AGENT LOGGERS
query_agent_logger = logging.getLogger("agent_query_classifier")
query_agent_logger.setLevel(logging.INFO)
query_agent_logger.addHandler(get_rotating_handler("agent_query_classifier.log"))

retriever_agent_logger = logging.getLogger("agent_retriever")
retriever_agent_logger.setLevel(logging.INFO)
retriever_agent_logger.addHandler(get_rotating_handler("agent_retriever.log"))

synth_agent_logger = logging.getLogger("agent_synthesizer")
synth_agent_logger.setLevel(logging.INFO)
synth_agent_logger.addHandler(get_rotating_handler("agent_synthesizer.log"))


# RAG PIPELINE LOGGER
pipeline_logger = logging.getLogger("rag_pipeline")
pipeline_logger.setLevel(logging.INFO)
pipeline_logger.addHandler(get_rotating_handler("rag_pipeline.log"))


# Helper: create request ID
def new_request_id():
    return str(uuid.uuid4())

# -------------------------------------------------------------------
# Compatibility functions for existing imports
# -------------------------------------------------------------------

def setup_logging():
    """Kept for backward compatibility. Logging already initialized globally."""
    pass


def get_logger(name: str):
    """Return a logger by name."""
    return logging.getLogger(name)
