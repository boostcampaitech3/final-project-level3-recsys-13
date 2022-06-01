import os
from dotenv import load_dotenv

load_dotenv()

PROTOCOL = os.getenv("PROTOCOL", "http")
API_URL = os.getenv("API_URL")
API_DEST = PROTOCOL + "://" + API_URL