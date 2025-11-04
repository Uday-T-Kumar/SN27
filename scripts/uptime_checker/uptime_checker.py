#!/usr/bin/env python3
import avro.schema as schema
import io
import configparser
import json
import logging
import requests
import time
from avro.io import BinaryEncoder, DatumWriter
from datetime import datetime, timezone
from google.api_core.exceptions import NotFound
from google.cloud.pubsub import PublisherClient
from google.pubsub_v1.types import Encoding
from requests.exceptions import HTTPError


# Module to run curl requests on IP:Port and report back if they are reachable or not
def check_miner_ip(final_url):
    result = 5
    try:
        logger.info("module start: check miner ip" )
        #print("func check miner ip start")
        response = requests.get(final_url, timeout=0.4) #0.5,1,2
        logger.info(response)
        logger.info(response.status_code == 200)
        result = 4
        logger.info("1 - 200 - Success")
    except HTTPError as http_err:              # Handling the HTTP errors
        if http_err.response.status_code ==404:
            logger.warning("1a - 404 - URL Didn't work")
            result = 6                        # Marking them as true because it still 
        elif http_err.response.status_code ==500:
            logger.warning("1b - Server error")
            result = 6
    except requests.exceptions.Timeout:        # Handling timeouts
        logger.warning("2- The request timed out!")
        result = 6
    except requests.exceptions.ConnectionError as e:
        if e.args and isinstance(e.args[0], tuple) and isinstance(e.args[0][1], ConnectionRefusedError):
            errno = e.args[0][1].errno
            logger.eror(f"3a -ConnectionError: Errno = {errno}")
            result = 5
        else:
            logger.error(f"3b -ConnectionError: {e}")
            result = 5
            #print(f"ErrNo. - {errno}")
    except requests.exceptions.RequestException as e:  #Handling other request errors
        logger.error(f"4 -An error occurred: {e}")
        result = 5
    except Exception as e:
        logger.error(f"5- Unexpected error {e}")
        result = 5
    
    if(result == 4 or result == 6):
        return True
    else: 
        return False


#def pubsub_subs(topic_path, bout, record):
def pubsub_subs(topic_path, bout, record):
    '''
    project_id = config['gcloud']['project_id']
    topic_id = config['gcloud']['topic_id']
    avsc_file = config['gcloud']['avsc_file']
    publisher_client = PublisherClient()
    topic_path = publisher_client.topic_path(project_id, topic_id)

    # Prepare to write Avro records to the binary output stream.
    with open(avsc_file, "rb") as file:
        avro_schema = schema.parse(file.read())
        writer = DatumWriter(avro_schema)
        bout = io.BytesIO()
    '''
    try:
        # Get the topic encoding type.
        topic = publisher_client.get_topic(request={"topic": topic_path})
        encoding = topic.schema_settings.encoding

        # Encode the data according to the message serialization type.
        if encoding == Encoding.BINARY:
            encoder = BinaryEncoder(bout)
            writer.write(record, encoder)
            data = bout.getvalue()
            logger.debug(f"Preparing a binary-encoded message:\n{data.decode()}")
        elif encoding == Encoding.JSON:
            data_str = json.dumps(record)
            logger.debug(f"Preparing a JSON-encoded message:\n{data_str}")
            data = data_str.encode("utf-8")
        else:
            logger.debug(f"No encoding specified in {topic_path}. Abort.")
            exit(0)

        future = publisher_client.publish(topic_path, data)
        logger.info(f"Published message ID: {future.result()}")
    except NotFound:
        logger.debug(f"{topic_id} not found.")



if __name__ == '__main__':
    ## Perf testing
    start_time = time.perf_counter()
    local_time = time.localtime()
    formatted_time = time.strftime("%H:%M:%S", local_time)
    print(f"Start time:{formatted_time}, {start_time:.6f} seconds")

    #Config file
    config = configparser.ConfigParser()
    config.read('/home/sn27/SN27/scripts/uptime_checker/uptime_check_config.ini')

    #main
    #logging
    #logging.basicConfig(level=getattr(logging,config['Log']['level']),
    logging.basicConfig(level='INFO',
        filename=config['Log']['log_file'],
        encoding=config['Log']['log_encoding'],
        filemode=config['Log']['log_filemode'],
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging,config['Log']['level']))

    # print("***** Miner Uptime check start ******")
    logger.info("-----*****  Miner Uptime check start  ******-----")
    logger.info(f"Start time:{formatted_time}, {start_time:.6f} seconds")


    project_id = config['gcloud']['project_id']
    topic_id = config['gcloud']['topic_id']
    avsc_file = config['gcloud']['avsc_file']
    publisher_client = PublisherClient()
    topic_path = publisher_client.topic_path(project_id, topic_id)

    # Prepare to write Avro records to the binary output stream.
    with open(avsc_file, "rb") as file:
        avro_schema = schema.parse(file.read())
    writer = DatumWriter(avro_schema)
    bout = io.BytesIO()


    #final_url = [] #multiprocessing
    # Use the data from metagraf, check their connectivity & report to gcloud PubSub
    with open("metagrafdata.txt", "r") as file_object:
        lines = file_object.readlines()
        for line in lines:
            parts = line.split(' ')
            ip = parts[1]
            port = parts[2]
            hkey = parts[3]
            logger.info(f"IP: {ip}, Port: {port}, hkey: {hkey}") #,str(ip),"Port:",str(port),"hkey:",str(hkey))
            final_url = "http://"+ip+":"+port
            #final_url.append = str("http://"+ip+":"+port) #multiprocessing
            logger.info(final_url)

            resp=check_miner_ip(final_url)  #use the final URL to invoke function to make a curl request on IP:port; Response is in the form of true/false
            logger.info(resp)
            #print(resp)
            reachable = resp
            datetime = datetime.now(timezone.utc)
            current_datetime = datetime.strftime("%Y-%m-%d, %H:%M:%S")

            # Prepare some data using a Python dictionary that matches the Avro schema
            record = {"Time": current_datetime, "ID": hkey, "Reachable": reachable}
            pubsub_subs(topic_path, bout, record)
            #pubsub_subs(record)

    '''
    with multiprocessing.Pool(processes=4) as pool: #use a pool of 4 worker processes
        resp=pool.map(check_miner_ip, final_url)
        print(resp)
    '''


    # Record the end time
    end_time = time.perf_counter()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Log/Print the execution time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    logger.info(f"Execution time: {elapsed_time:.6f} seconds")
