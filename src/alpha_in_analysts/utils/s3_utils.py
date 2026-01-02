import pandas as pd
import logging
from typing import List, Dict
import pickle
import boto3
from botocore.client import BaseClient
import io
import os

logger = logging.getLogger(__name__)

class s3Utils:
    @staticmethod
    def push_object_to_s3_parquet(object_to_push:pd.DataFrame,
                                  path:str)->None:
        """
        Method to upload a dataframe to s3 in parquet format at the specified path
        :param object_to_push:
        :param path:
        :return: None
        """
        if not isinstance(object_to_push, pd.DataFrame):
            logger.error("object_to_push must be a pd.DataFrame")
            raise ValueError("object_to_push must be a pd.DataFrame")
        if not isinstance(path, str):
            logger.error("path must be a str")
            raise ValueError("path must be a str")

        object_to_push.to_parquet(
            path=path,
            engine="pyarrow"
        )
        return

    @staticmethod
    def push_objects_to_s3_parquet(objects_dct:dict)->None:
        """
        Method to upload several dfs to s3 in parquet format at the given paths
        :param objects_dct: objects as values and paths as keys
        :return: None
        """
        if not isinstance(objects_dct, dict):
            logger.error("objects_dct must be a dict.")
            raise ValueError("objects_dct must be a dict.")
        for path, obj in objects_dct.items():
            s3Utils.push_object_to_s3_parquet(object_to_push=obj,
                                              path=path)
        return

    @staticmethod
    def pull_parquet_file_from_s3(path:str)->pd.DataFrame:
        if not isinstance(path, str):
            logger.error("path must be a str.")
            raise ValueError("path must be a str.")
        res = pd.read_parquet(path=path,
                              engine="pyarrow"
                              )
        return res

    @staticmethod
    def pull_parquet_files_from_s3(paths:List[str])->dict:
        """
        Given a list of s3 files paths returns dfs in a dict
        :param paths: list of s3 paths
        :return: dict of dfs with keys as names of the s3 object and the dfs as values
        """
        if not isinstance(paths,list):
            logger.error("paths must be a list.")
            raise ValueError("paths must be a list.")
        for path in paths:
            if not isinstance(path, str):
                logger.error("all paths must be strings.")
                raise ValueError("all paths must be strings.")

        res_dct = {}
        for path in paths:
            name = path.split("/")
            name = name[-1]
            name = name.split(".")
            name = name[0]
            res = s3Utils.pull_parquet_file_from_s3(path=path)
            res_dct[name] = res

        return res_dct

    @staticmethod
    def replace_existing_files_in_s3(s3: BaseClient,
                                     bucket_name: str,
                                     files_dct: dict
                                     ) -> None:

        """
        Given a dict with the file names (as the ones in s3) as keys and the file content as values,
        replace the existing files in s3 with the new content.
        :return:
        """
        # Check data types of inputs
        if not hasattr(s3, 'put_object'):
            logger.error("s3 must be a boto3 BaseClient instance.")
            raise ValueError("s3 must be a boto3 BaseClient instance.")
        if not isinstance(bucket_name, str):
            logger.error("bucket_name must be a string.")
            raise ValueError("bucket_name must be a string.")
        if not isinstance(files_dct, dict):
            logger.error("files_dct must be a dictionary.")
            raise ValueError("files_dct must be a dictionary with file names as keys and file content as values.")

        # Check that the file names (keys of the dict) are all present on s3
        for file_name in files_dct.keys():
            try:
                s3.head_object(Bucket=bucket_name, Key=file_name)
            except Exception as e:
                logger.error(f"File {file_name} does not exist in S3 bucket {bucket_name}: {e}")
                raise ValueError(f"File {file_name} does not exist in S3 bucket {bucket_name}.")

        # Upload new versions first
        for file_name, content in files_dct.items():
            ext = file_name.split('.')[-1]

            if ext == "parquet":
                buffer = io.BytesIO()
                content.to_parquet(buffer, index=True)
                buffer.seek(0)
                body = buffer

            elif ext == "pkl":
                body = pickle.dumps(content)

            else:
                raise ValueError(f"Unsupported extension: {file_name}")

            s3.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=body
            )
            logger.info(f"Uploaded new version of {file_name} to S3 bucket {bucket_name}.")

        # Delete all non-latest versions + delete markers
        for file_name in files_dct.keys():
            paginator = s3.get_paginator("list_object_versions")

            to_delete = []

            for page in paginator.paginate(Bucket=bucket_name, Prefix=file_name):

                for v in page.get("Versions", []):
                    if not v["IsLatest"]:
                        to_delete.append({
                            "Key": file_name,
                            "VersionId": v["VersionId"]
                        })

                for d in page.get("DeleteMarkers", []):
                    if not d["IsLatest"]:
                        to_delete.append({
                            "Key": file_name,
                            "VersionId": d["VersionId"]
                        })

            if to_delete:
                s3.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": to_delete}
                )
            logger.info(f"Deleted previous versions of {file_name} in S3 bucket {bucket_name}.")

    @staticmethod
    def get_credentials(return_bool:bool=False
                        )->Dict[str, str]:
        """
        Load credentials from .env .
        :param:
        - return_bool: If True, returns the credentials dict.
        :return: a dict containing the credentials.
        """
        creds = {}
        for key in ["KEY", "SECRET_KEY","REGION","OUTPUT_FORMAT"]:
            val = os.getenv(key)
            if val is not None:
                creds[key] = val

        required = ["KEY", "SECRET_KEY","REGION","OUTPUT_FORMAT"]
        missing = [k for k in required if k not in creds]

        if missing:
            logger.error(f"Missing credentials: {missing}")
            raise RuntimeError(f"Missing credentials: {missing}")

        if return_bool:
            return creds

    @staticmethod
    def connect_aws_s3(creds)->BaseClient:
        """
        Connect to AWS S3 using the loaded credentials.
        :return: boto3 S3 client.
        """

        s3 = boto3.client(
            's3',
            aws_access_key_id=creds["KEY"],
            aws_secret_access_key=creds["SECRET_KEY"],
            region_name=creds["REGION"]
        )
        return s3