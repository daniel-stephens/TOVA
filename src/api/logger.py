import logging

logger = logging.getLogger("TOVA_API")

if not logger.handlers:
    print("WARNING: 'TOVA_API' logger accessed before full configuration. Handlers might be missing.")