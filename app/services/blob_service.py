import requests
import io
import joblib
from app.utils.logger import get_logger

logger = get_logger(__name__)

def load_joblib_from_url(url: str):
    try:
        logger.info(f"Downloading joblib file from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        model_data = io.BytesIO(response.content)
        obj = joblib.load(model_data)
        logger.info("Successfully loaded joblib file")
        return obj
    except requests.RequestException as e:
        logger.error(f"Failed to download file from URL: {e}")
        raise Exception(f"Failed to download file from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to load joblib from downloaded data: {e}")
        raise Exception(f"Failed to load joblib from downloaded data: {e}")
