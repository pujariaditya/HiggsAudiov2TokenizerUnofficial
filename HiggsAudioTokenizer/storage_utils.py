# Copyright 2025 HiggsAudiov2-Unofficial Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cloud storage utilities for HiggsAudiov2 (Unofficial Implementation).

Provides unified interface for local filesystem, Google Cloud Storage (GCS),
and AWS S3 storage backends.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Union, Any
import time
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    import boto3
    HAS_S3 = True
except ImportError:
    HAS_S3 = False


# Cached client management
@lru_cache(maxsize=1)
def _get_gcs_client():
    """Get cached GCS client."""
    if not HAS_GCS:
        raise ImportError("google-cloud-storage is required for GCS support")
    try:
        return gcs.Client()
    except Exception as e:
        logger.warning(f"Failed to initialize GCS client: {e}")
        raise


@lru_cache(maxsize=1)
def _get_s3_client():
    """Get cached S3 client."""
    if not HAS_S3:
        raise ImportError("boto3 is required for S3 support")
    try:
        return boto3.client('s3')
    except Exception as e:
        logger.warning(f"Failed to initialize S3 client: {e}")
        raise


# Initialize clients with proper error handling
try:
    _gcs_client = _get_gcs_client() if HAS_GCS else None
except Exception:
    logger.warning("Failed to initialize GCS client. GCS operations will require runtime initialization.")
    _gcs_client = None

try:
    _s3_client = _get_s3_client() if HAS_S3 else None
except Exception:
    logger.warning("Failed to initialize S3 client. S3 operations will require runtime initialization.")
    _s3_client = None


class CloudPath:
    """Unified path interface for local, GCS, and S3 storage."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        self._parse_path()
    
    def _parse_path(self):
        """Parse path to determine storage type."""
        if self.path.startswith('gs://'):
            self.storage_type = 'gcs'
            self.bucket_name = self.path.split('/')[2]
            self.blob_name = '/'.join(self.path.split('/')[3:])
        elif self.path.startswith('s3://'):
            self.storage_type = 's3'
            self.bucket_name = self.path.split('/')[2]
            self.key = '/'.join(self.path.split('/')[3:])
        else:
            self.storage_type = 'local'
            self.local_path = Path(self.path)
    
    @property
    def is_cloud(self) -> bool:
        """Check if path is a cloud path."""
        return self.storage_type in ['gcs', 's3']
    
    @property
    def is_local(self) -> bool:
        """Check if path is a local path."""
        return self.storage_type == 'local'
    
    def __str__(self):
        return self.path
    
    def __repr__(self):
        return f"CloudPath('{self.path}')"
    
    @property
    def name(self) -> str:
        """Get the name of the file/blob."""
        if self.storage_type == 'gcs':
            return self.blob_name.split('/')[-1]
        elif self.storage_type == 's3':
            return self.key.split('/')[-1]
        else:
            return self.local_path.name
    
    def is_dir(self) -> bool:
        """Check if path represents a directory."""
        if self.storage_type == 'local':
            return self.local_path.is_dir()
        else:
            # For cloud paths, check if it ends with /
            return self.path.endswith('/')
    
    def download_to(self, local_path: Union[str, Path]):
        """Download file from cloud to local path."""
        local_path = Path(local_path)
        
        if self.storage_type == 'gcs':
            if not HAS_GCS:
                raise ImportError("google-cloud-storage is required for GCS operations")
            client = _get_gcs_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.blob_name)
            blob.download_to_filename(str(local_path))
        elif self.storage_type == 's3':
            if not HAS_S3:
                raise ImportError("boto3 is required for S3 operations")
            s3 = _get_s3_client()
            s3.download_file(self.bucket_name, self.key, str(local_path))
        else:
            # Local to local copy
            shutil.copy2(self.local_path, local_path)
    
    def exists(self) -> bool:
        """Check if path exists."""
        if self.storage_type == 'local':
            return self.local_path.exists()
        elif self.storage_type == 'gcs':
            if not HAS_GCS:
                return False
            try:
                client = _get_gcs_client()
                bucket = client.bucket(self.bucket_name)
                blob = bucket.blob(self.blob_name)
                return blob.exists()
            except:
                return False
        elif self.storage_type == 's3':
            if not HAS_S3:
                return False
            try:
                s3 = _get_s3_client()
                s3.head_object(Bucket=self.bucket_name, Key=self.key)
                return True
            except:
                return False


def is_cloud_path(path: Union[str, Path]) -> bool:
    """Check if a path is a cloud storage path."""
    path_str = str(path)
    return path_str.startswith(('gs://', 's3://'))


# GCS upload utility (required by upload_file)
def gcs_upload(local_path: Union[str, Path], bucket_name: str, blob_name: str,
               retry_count: int = 3):
    """Upload a file to GCS with retry logic."""
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    for attempt in range(retry_count):
        try:
            blob.upload_from_filename(str(local_path))
            return
        except Exception as e:
            if attempt == retry_count - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff


# S3 upload utility (required by upload_file)
def s3_upload(local_path: Union[str, Path], bucket_name: str, key: str,
              retry_count: int = 3):
    """Upload a file to S3 with retry logic."""
    s3 = _get_s3_client()
    
    for attempt in range(retry_count):
        try:
            s3.upload_file(str(local_path), bucket_name, key)
            return
        except Exception as e:
            if attempt == retry_count - 1:
                raise e
            time.sleep(2 ** attempt)


# High-level utilities
def upload_file(local_path: Union[str, Path], cloud_path: Union[str, Path],
                retry_count: int = 3):
    """Upload a file to cloud storage (GCS or S3)."""
    local_path = Path(local_path)
    cloud_path = CloudPath(cloud_path)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    if cloud_path.is_local:
        # Local to local copy
        Path(cloud_path.path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, cloud_path.path)
    elif cloud_path.storage_type == 'gcs':
        gcs_upload(local_path, cloud_path.bucket_name, cloud_path.blob_name, retry_count)
    elif cloud_path.storage_type == 's3':
        s3_upload(local_path, cloud_path.bucket_name, cloud_path.key, retry_count)


def torch_save_to_cloud(save_obj: Any, save_path: Union[str, Path], compress: bool = True) -> None:
    """Save a PyTorch object directly to cloud storage.
    
    Args:
        save_obj: Object to save (usually model state dict or checkpoint)
        save_path: Path to save in cloud storage
        compress: Whether to use compression. Default: True
    """
    import torch
    
    if not is_cloud_path(save_path):
        raise ValueError("save_path must be a cloud path (gs:// or s3://)")
    
    # Save to a temporary local file and then upload
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        torch.save(save_obj, tmp.name, _use_new_zipfile_serialization=compress)
        tmp_path = tmp.name
    
    try:
        upload_file(tmp_path, save_path)
    except Exception as e:
        raise RuntimeError(f"Error saving to cloud storage: {e}")
    finally:
        Path(tmp_path).unlink()


# Compatibility aliases and functions for utils.py
GSPath = CloudPath  # Alias for backward compatibility


def is_gcs_path(path: Union[str, Path]) -> bool:
    """Check if a path is a Google Cloud Storage path.
    
    Args:
        path: Path to check
        
    Returns:
        True if path starts with gs://
    """
    return str(path).startswith('gs://')