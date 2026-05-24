import requests
import io
import json
import joblib
from typing import Optional, Dict, Any, List, Tuple
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
    ) -> Optional[tuple[bytes, str]]:
        """Download latest model and return (bytes, blob_name) tuple."""
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
                return (model_bytes, latest_blob_name)
            else:
                blob_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{latest_blob_name}"
                logger.info(f"Attempting public URL download: {blob_url}")
                response = requests.get(blob_url, timeout=30)
                response.raise_for_status()
                logger.info(f"Successfully downloaded {len(response.content)} bytes via public URL")
                return (response.content, latest_blob_name)
                
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
    ) -> Optional[tuple[bytes, str]]:
        """Download newest versioned model and return (bytes, blob_name) tuple."""
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
            return (model_bytes, newest_blob.name)
            
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

    def download_specialty_models(
        self,
        environment: str = "homolog",
    ) -> Dict[str, Tuple[bytes, str]]:
        """
        Download all per-specialty models from blob storage.

        Looks for blobs matching the new path convention:
            {environment}/{specialty}/model_latest.joblib

        Returns:
            Dict {SPECIALTY_GROUP: (model_bytes, blob_name)}
        """
        if not self.blob_service_client:
            logger.error("Blob service client not initialized for listing specialty models")
            return {}

        prefix = f"{environment}/"

        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            models: Dict[str, Tuple[bytes, str]] = {}

            for blob in container_client.list_blobs(name_starts_with=prefix):
                # Pattern: {environment}/{specialty}/model_latest.joblib (exactly 3 parts)
                parts = blob.name.split("/")
                if (
                    len(parts) == 3
                    and parts[0] == environment
                    and parts[2] == "model_latest.joblib"
                ):
                    specialty_key = parts[1].upper()

                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container_name, blob=blob.name
                    )
                    data = blob_client.download_blob().readall()
                    models[specialty_key] = (data, blob.name)
                    logger.info(
                        f"Downloaded specialty model: {blob.name} "
                        f"({len(data)/(1024*1024):.2f} MB) → '{specialty_key}'"
                    )

            logger.info(f"Downloaded {len(models)} specialty model(s): {list(models.keys())}")
            return models

        except Exception as e:
            logger.error(f"Error downloading specialty models: {str(e)}")
            return {}

    def download_default_model(
        self,
        environment: str = "homolog",
    ) -> Optional[Tuple[bytes, str]]:
        """
        Download the default/fallback model from the root environment folder.

        Attempts ``{environment}/model_latest.joblib`` (the old single-model path).
        Returns None if no such file exists — callers must handle gracefully.

        Returns:
            (model_bytes, blob_name) tuple or None.
        """
        blob_path = f"{environment}/model_latest.joblib"
        try:
            if not self.blob_service_client:
                logger.warning("Blob service client not initialized; cannot download default model")
                return None

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            logger.info(
                f"Default model downloaded: {blob_path} ({len(data)/(1024*1024):.2f} MB)"
            )
            return (data, blob_path)

        except AzureError as e:
            logger.info(f"Default model not found at '{blob_path}': {str(e)}")
            return None
        except Exception as e:
            logger.info(f"Could not download default model from '{blob_path}': {str(e)}")
            return None

    def download_thresholds(
        self,
        environment: str = "homolog",
    ) -> Dict[str, float]:
        """
        Download the consolidated thresholds JSON from blob storage.

        Returns:
            Dict {SPECIALTY_GROUP: threshold_float}
        """
        blob_path = f"{environment}/thresholds/thresholds_latest.json"
        try:
            if not self.blob_service_client:
                logger.error("Blob service client not initialized")
                return {}

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            thresholds = json.loads(data.decode("utf-8"))
            logger.info(f"Downloaded thresholds for {len(thresholds)} specialties")
            return thresholds

        except AzureError as e:
            logger.warning(f"Could not download thresholds from {blob_path}: {str(e)}")
            return {}
        except Exception as e:
            logger.warning(f"Error downloading thresholds from {blob_path}: {str(e)}")
            return {}

    def download_stats_parquet(
        self,
        environment: str = "homolog",
        filename: str = "patient_stats_latest.parquet",
    ) -> Optional[bytes]:
        """
        Download a precomputed stats parquet file from blob storage.

        Path: {environment}/stats/{filename}

        Args:
            environment: Environment folder.
            filename: Name of the stats parquet file.

        Returns:
            Raw bytes of the parquet file, or None if not found.
        """
        blob_path = f"{environment}/stats/{filename}"
        try:
            if not self.blob_service_client:
                logger.warning("Blob service client not initialized for stats download")
                return None

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            logger.info(
                f"Downloaded stats parquet: {blob_path} ({len(data) / 1024:.1f} KB)"
            )
            return data

        except AzureError as e:
            logger.info(f"Stats parquet not found at '{blob_path}': {str(e)}")
            return None
        except Exception as e:
            logger.info(f"Could not download stats parquet from '{blob_path}': {str(e)}")
            return None

    def download_cluster_artifact(
        self,
        environment: str = "develop",
    ) -> Optional[bytes]:
        """
        Download the K-Means patient cluster artifact produced by noshow_lib v0.4.0.

        Path: {environment}/artifacts/kmeans_cluster_patient_latest.joblib

        Returns:
            Raw bytes of the joblib artifact, or None if not found.
        """
        blob_path = f"{environment}/artifacts/kmeans_cluster_patient_latest.joblib"
        try:
            if not self.blob_service_client:
                logger.warning("Blob service client not initialized for cluster artifact download")
                return None

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_path
            )
            data = blob_client.download_blob().readall()
            logger.info(
                f"Downloaded K-Means cluster artifact: {blob_path} ({len(data) / 1024:.1f} KB)"
            )
            return data

        except AzureError as e:
            logger.info(f"Cluster artifact not found at '{blob_path}': {str(e)}")
            return None
        except Exception as e:
            logger.info(f"Could not download cluster artifact from '{blob_path}': {str(e)}")
            return None

