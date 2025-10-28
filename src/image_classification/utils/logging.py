import logging
import os


def init_logging(log_dir: str, *dirs: str, log_filename: str = 'log.txt') -> logging.Logger:
    """
    Initialize logging settings.

    Args:
        log_dir (str): Directory where logs will be saved.
        *dirs (str): Additional directories to create within the log directory.
        log_filename (str): Name of the log file. Defaults to 'log.txt'.
    """
    # Create the log directory if it does not exist
    log_file = os.path.join(log_dir, *dirs, log_filename)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set file logging style
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    # Set console logging style
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info(f"Logging initialized. Logs will be saved to \n\t{log_file}")
    return logger
