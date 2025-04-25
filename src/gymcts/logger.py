import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)]
)

log = logging.getLogger("rich")

banner_sw = f"""

      ▟█████▛▜█▙▟█▛ ▟█▙  ▟██▛▟█████▛████▛▟████▛
     ▟█▛      ▜██▛ ▟█▛██▛██▛▟█▛     ▟█▛  ▜███▙ 
    ▟█▛ ▟█▛   ▟█▛ ▟█▛   ▟█▛▟█▛     ▟█▛      ▟█▛ 
    ▜████▛   ▟█▛ ▟█▛   ▟█▛ ▜████▛ ▟█▛  ▟████▛  
                                           
"""


if __name__ == '__main__':
    log.debug("Hello, World!")
    log.info("Hello, World!")
    log.error("Hello, World!")
    log.warning("Hello, World!")
    log.critical("Hello, World!")
