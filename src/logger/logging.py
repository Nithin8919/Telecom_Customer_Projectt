import logging
import os
from datetime import datetime

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")

# Create log directory if it doesn't exist
os.makedirs(log_path, exist_ok=True)

LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    filename=LOG_FILEPATH,
                    filemode='a',  # Append mode; change to 'w' for overwrite
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

if __name__ == '__main__':
    logging.info("This is a test log message")
