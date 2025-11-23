import requests
import io
import joblib
from app.utils.logger import get_logger

logger = get_logger(__name__)

def load_joblib_from_url(url: str):
    """Download and load a joblib file from Azure Blob Storage."""
    try:
        logger.info(f"Downloading joblib file from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        logger.info(f"Downloaded file content-type: {content_type}, size: {len(response.content)} bytes")
        
        model_data = io.BytesIO(response.content)
        obj = joblib.load(model_data)
        
        logger.info(f"Successfully loaded joblib file. Object type: {type(obj).__name__}")
        return obj
        
    except requests.Timeout:
        logger.error(f"Request timeout while downloading from URL: {url}")
        raise Exception(f"Request timeout while downloading from URL: {url}")
    except requests.RequestException as e:
        logger.error(f"Failed to download file from URL: {e}")
        raise Exception(f"Failed to download file from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to load joblib from downloaded data: {e}")
        raise Exception(f"Failed to load joblib from downloaded data: {e}")
