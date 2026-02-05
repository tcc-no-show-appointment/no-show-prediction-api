import requests
import io
import joblib
from typing import Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from app.utils.logger import get_logger

logger = get_logger(__name__)

def load_joblib_from_url(url: str):
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


class BlobStorageClient:
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: str = "devconteiner"
    ):
        self.container_name = container_name
        self.account_name = account_name or "devstoragecenter"
        self.blob_service_client = None
        
        try:
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Blob service client initialized with connection string")
            elif account_name and account_key:
                account_url = f"https://{account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=account_key
                )
                logger.info("Blob service client initialized with account credentials")
            else:
                logger.warning("No Azure credentials provided. Using public URL access.")
        except Exception as e:
            logger.error(f"Failed to initialize Blob service client: {str(e)}")
            self.blob_service_client = None
    
    def download_latest_model(
        self,
        environment: str = "homolog",
        base_name: str = "model"
    ) -> Optional[bytes]:
        latest_blob_name = f"{environment}/{base_name}_latest.joblib"
        
        try:
            logger.info(f"Downloading latest model from: {latest_blob_name}")
            
            if self.blob_service_client:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=latest_blob_name
                )
                
                download_stream = blob_client.download_blob()
                model_bytes = download_stream.readall()
                logger.info(f"Successfully downloaded {len(model_bytes)} bytes using Azure SDK")
                return model_bytes
            else:
                blob_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{latest_blob_name}"
                logger.info(f"Attempting public URL download: {blob_url}")
                response = requests.get(blob_url, timeout=30)
                response.raise_for_status()
                logger.info(f"Successfully downloaded {len(response.content)} bytes via public URL")
                return response.content
                
        except AzureError as e:
            logger.error(f"Azure error downloading latest model: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error downloading latest model: {str(e)}")
            return None
    
    def download_newest_versioned_model(
        self,
        environment: str = "homolog",
        base_name: str = "model"
    ) -> Optional[bytes]:
        if not self.blob_service_client:
            logger.error("Blob service client not initialized for listing")
            return None
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_prefix = f"{environment}/{base_name}_"
            
            logger.info(f"Listing versioned models with prefix: {blob_prefix}")
            
            versioned_blobs = []
            for blob in container_client.list_blobs(name_starts_with=blob_prefix):
                if blob.name.endswith('.joblib') and '_latest.joblib' not in blob.name:
                    versioned_blobs.append(blob)
            
            if not versioned_blobs:
                logger.warning(f"No versioned models found in {environment} environment")
                return None
            
            versioned_blobs.sort(key=lambda b: b.last_modified, reverse=True)
            newest_blob = versioned_blobs[0]
            
            logger.info(f"Found newest versioned model: {newest_blob.name} (modified: {newest_blob.last_modified})")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=newest_blob.name
            )
            
            download_stream = blob_client.download_blob()
            model_bytes = download_stream.readall()
            logger.info(f"Successfully downloaded {len(model_bytes)} bytes from versioned model")
            return model_bytes
            
        except Exception as e:
            logger.error(f"Error downloading newest versioned model: {str(e)}")
            return None
    
    def download_config_file(
        self,
        folder: str = "model_configuration",
        filename: str = "prod.yaml"
    ) -> Optional[str]:
        """
        Download configuration file from blob storage.
        
        Args:
            folder: Folder name in blob storage (default: 'model_configuration')
            filename: Name of the config file (default: 'prod.yaml')
            
        Returns:
            Configuration file content as string, or None if download fails
        """
        blob_path = f"{folder}/{filename}"
        
        try:
            logger.info(f"Downloading config file from: {blob_path}")
            
            if self.blob_service_client:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_path
                )
                
                download_stream = blob_client.download_blob()
                config_bytes = download_stream.readall()
                config_content = config_bytes.decode('utf-8')
                logger.info(f"Successfully downloaded config file ({len(config_bytes)} bytes) using Azure SDK")
                return config_content
            else:
                blob_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_path}"
                logger.info(f"Attempting public URL download: {blob_url}")
                response = requests.get(blob_url, timeout=30)
                response.raise_for_status()
                config_content = response.text
                logger.info(f"Successfully downloaded config file ({len(config_content)} bytes) via public URL")
                return config_content
                
        except AzureError as e:
            logger.error(f"Azure error downloading config file: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error downloading config file from {blob_path}: {str(e)}")
            return None
