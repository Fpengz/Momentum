# logger_utils.py
import logging

def setup_logger(verbose: bool = True, name: str = "pipeline") -> logging.Logger:
    """
    Configure and return a logger.

    Parameters
    ----------
    verbose : bool
        If True, INFO logs are shown. Otherwise, only WARNING+.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()  # avoid duplicates when re-importing

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # set log level
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger
