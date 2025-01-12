from pathlib import Path
import os

from dotenv import load_dotenv
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

load_dotenv(Path("..", "..", ".env"))

ml_client = MLClient.from_config(
    credential=AzureCliCredential(), file_name="config.json"
)

iris_data_unlabeled = Data(
    path="azureml://datastores/workspaceblobstore/paths/template_data/scoring/",
    type=AssetTypes.URI_FOLDER,
    description="Unlabeled iris dataset for use in Azure ML Template Project",
    name="template_data_iris_unlabeled",
)

ml_client.data.create_or_update(iris_data_unlabeled)
