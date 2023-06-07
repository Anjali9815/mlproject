import logging
import os, sys
from datetime import datetime


Log_file = f"{datetime.now().strftime('%m%d%Y')}"
log_path = os.path.join(os.getcwd(), "logs", Log_file)

os.makedirs(log_path, exist_ok = True)
Log_file = f"{datetime.now().strftime('%m%d%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(log_path, Log_file)

logging.basicConfig(

    filename= log_file_path,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s %(pathname)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging Started")
