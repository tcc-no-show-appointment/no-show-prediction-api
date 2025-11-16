import requests
import io
import pandas as pd
import pickle
from app.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv_from_url(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Downloading CSV from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        logger.info(f"Successfully loaded CSV with {len(df)} rows")
        return df
    except requests.RequestException as e:
        logger.error(f"Failed to download CSV from URL: {e}")
        raise Exception(f"Failed to download CSV from URL: {e}")
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV data: {e}")
        raise Exception(f"Failed to parse CSV data: {e}")


def load_pickle_from_url(url: str):
    try:
        logger.info(f"Downloading pickle file from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        model_data = io.BytesIO(response.content)
        obj = pickle.load(model_data)
        logger.info("Successfully loaded pickle file")
        return obj
    except requests.RequestException as e:
        logger.error(f"Failed to download file from URL: {e}")
        raise Exception(f"Failed to download file from URL: {e}")
    except pickle.PickleError as e:
        logger.error(f"Failed to load pickle from downloaded data: {e}")
        raise Exception(f"Failed to load pickle from downloaded data: {e}")
