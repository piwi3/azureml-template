from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import mlflow
from mlflow.tracking import MlflowClient
import os

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

parser = argparse.ArgumentParser("model_selection")
parser.add_argument("--train_input_path", type=str, help="Input path of the train set")
parser.add_argument("--test_input_path", type=str, help="Input path of the train set")
parser.add_argument("--model_input_path", type=str, help="Input path of the model")
parser.add_argument("--params_input_path", type=str, help="Input path of parameters to use")

args = parser.parse_args()

train_dat = pd.read_parquet(select_first_file(args.train_input_path))
test_dat = pd.read_parquet(select_first_file(args.test_input_path))

input_model = mlflow.sklearn.load_model(select_first_file(args.model_input_path))


def split(x):
    return x.pop("species").values, x.drop(columns=["fold"]).to_numpy()


train_y, train_X = (
    train_dat.pop("species").values,
    train_dat.drop(columns=["fold"]).to_numpy(),
)

test_y, test_X = (test_dat.pop("species").values, test_dat.to_numpy())

with Path(select_first_file(args.params_input_path)).open(mode="r") as f:
    params = json.load(f)
print(params)

model = mlflow.sklearn.load_model(Path(select_first_file(args.model_input_path)))
output_model = clone(input_model)

output_model.fit(train_X, train_y)

y_pred = model.predict(test_X)

accuracy = accuracy_score(test_y, y_pred)

print(f"C: {output_model.get_params()['C']}")
print(f"Accuracy: {accuracy}")
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(output_model, "model")

mlflow_run = mlflow.active_run()
run_id = mlflow_run.info.run_id
model_path = "model"
model_uri = "runs:/{}/{}".format(run_id, model_path)
mlflow.register_model(model_uri, "template_model")
