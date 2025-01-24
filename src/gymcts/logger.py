import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)]
)

log = logging.getLogger("rich")


if __name__ == '__main__':
    log.debug("Hello, World!")
    log.info("Hello, World!")
    log.error("Hello, World!")
    log.warning("Hello, World!")
    log.critical("Hello, World!")
