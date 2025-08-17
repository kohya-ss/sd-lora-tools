import argparse
import logging
import sys
from typing import Optional


def add_logging_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--log_level", default="INFO", help="Set the logging level")
    parser.add_argument("--console_log_file", help="File to write console logs to")
    parser.add_argument("--console_log_simple", action="store_true", help="Use simple console logging format")


def setup_logging(args: Optional[argparse.Namespace] = None, log_level: Optional[str] = None, reset: bool = False):
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.log_level
    if log_level is None:
        log_level = "INFO"
    log_level_value: int = getattr(logging, log_level)

    # check if rich is installed
    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler  # type: ignore
                from rich.console import Console  # type: ignore
                from rich.logging import RichHandler  # type: ignore

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            # handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level_value)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)
