import logging
import requests
import sys
import bittensor as bt
from requests.exceptions import HTTPError

if __name__ == '__main__':
    #Logging
    logging.basicConfig(level=logging.DEBUG,
        filename='/home/sn27/SN27/scripts/uptime_checker/metagraf.log',
        encoding='utf-8',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    '''   
    [Log]
    level = DEBUG
    log_file = uptime.log
    log_encoding = utf-8
    log_filemode = a
    '''
    # Retreiving info from metagraf
    try:
        m = bt.metagraph(netuid=15, network="test", sync=True)
        for uid, ax in enumerate(m.axons):
            with open("metagrafdata.txt", "a") as file:
                file.write(f"{uid}, {ax.ip} {ax.port} {m.hotkeys[uid]}\n")
                logger.info("Successfully retrieved metagraf data")
    except HTTPError as http_err:              # Handling the HTTP errors
        sys.exit(1)
        if http_err.response.status_code ==404:
            logger.error("1a - 404 - URL Didn't work")
        elif http_err.response.status_code ==500:
            logger.error("1b - Server error")
        elif http_err.response.status_code ==429:
            logger.error("1c - Too many requests")
    except requests.exceptions.Timeout:        # Handling timeouts
        logger.error("2- The request timed out!")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        sys.exit(1)
        if e.args and isinstance(e.args[0], tuple) and isinstance(e.args[0][1], ConnectionRefusedError):
            errno = e.args[0][1].errno
            logger.error(f"3a -ConnectionError: Errno = {errno}")
        else:
            logger.error(f"3b -ConnectionError: {e}")
    except requests.exceptions.RequestException as e:  #Handling other request errors
        logger.error(f"4 -An error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"5- Unexpected error {e}")
        sys.exit(1)
