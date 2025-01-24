from gymcts.logger import log


banner_sw = f"""
  ▟█████▛▜█▙▟█▛ ▟█▙   ▟█▛▟█████▛████▛▟████▛
 ▟█▛      ▜██▛ ▟█▛█████▛▟█▛     ▟█▛  ▜███▙ 
▟█▛ ▟█▛   ▟█▛ ▟█▛   ▟█▛▟█▛     ▟█▛      ▟█▛ 
▜████▛   ▟█▛ ▟█▛   ▟█▛ ▜████▛ ▟█▛  ▟████▛  
                                       
"""

if __name__ == '__main__':
    log.debug("Hello, World!")
    log.info("Hello, World!")
    log.error("Hello, World!")
    log.warning("Hello, World!")
    print(banner_sw)
