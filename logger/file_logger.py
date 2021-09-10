import os
import logging
from pathlib import Path


class FileLogger:
    def __init__(self, log_path):
        parent_path = Path(os.path.realpath('__file__')).parent
        os.makedirs(os.path.join(parent_path, log_path), exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(parent_path, log_path, 'logs.txt'),
            level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    def debug(self, message):
        logging.debug(message)

    def info(self, message):
        logging.info(message)

    def error(self, message):
        logging.error(message)

    def warning(self, message):
        logging.warning(message)

    def critical(self, message):
        logging.critical(message)

    def disable(self):
        logging.disable()
