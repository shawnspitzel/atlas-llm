import io
from pathlib import Path

import boto3
import torch
import yaml
from botocore.exceptions import ClientError


S3_CONFIG_PATH = "src/configs/s3.yaml"


class S3Storage:
    _client = None
    _config = None

    @classmethod
    def load_config(cls) -> dict:
        if cls._config is not None:
            return cls._config

        config_path = Path(S3_CONFIG_PATH)
        if not config_path.exists():
            raise FileNotFoundError(
                f"S3 config not found at {S3_CONFIG_PATH}. "
                f"Create this file with s3-bucket, s3-region, and s3-prefix settings."
            )
        with open(config_path) as f:
            cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def get_client(cls):
        if cls._client is not None:
            return cls._client

        config = cls.load_config()
        region = config.get("s3-region", "us-east-1")

        cls._client = boto3.client("s3", region_name=region)
        return cls._client

    @classmethod
    def get_bucket(cls) -> str:
        config = cls.load_config()
        bucket = config.get("s3-bucket")
        if not bucket:
            raise ValueError(
                f"s3-bucket not set in {S3_CONFIG_PATH}."
            )
        return bucket

    @classmethod
    def get_key(cls, path: str) -> str:
        config = cls.load_config()
        prefix = config.get("s3-prefix", "")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{path}"

    @classmethod
    def save(cls, state_dict: dict, key: str) -> str:
        client = cls.get_client()
        bucket = cls.get_bucket()
        full_key = cls.get_key(key)

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        client.upload_fileobj(buffer, bucket, full_key)

        return f"s3://{bucket}/{full_key}"

    @classmethod
    def load(cls, key: str, device=None) -> dict:
        client = cls.get_client()
        bucket = cls.get_bucket()
        full_key = cls.get_key(key)

        buffer = io.BytesIO()
        client.download_fileobj(bucket, full_key, buffer)
        buffer.seek(0)
        if device is not None:
            state_dict = torch.load(buffer, map_location=device)
        else:
            state_dict = torch.load(buffer)
        return state_dict

    @classmethod
    def exists(cls, key: str) -> bool:
        client = cls.get_client()
        bucket = cls.get_bucket()
        full_key = cls.get_key(key)

        try:
            client.head_object(Bucket=bucket, Key=full_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    @classmethod
    def list_checkpoints(cls, prefix: str = "") -> list[str]:
        client = cls.get_client()
        bucket = cls.get_bucket()
        full_prefix = cls.get_key(prefix)

        checkpoints = []
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".pt"):
                        checkpoints.append(obj["Key"])
        return checkpoints

    @classmethod
    def delete(cls, key: str) -> None:
        client = cls.get_client()
        bucket = cls.get_bucket()
        full_key = cls.get_key(key)

        client.delete_object(Bucket=bucket, Key=full_key)
