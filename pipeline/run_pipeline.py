from azure.ai.ml import MLClient, Input, load_component
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.ml.dsl import pipeline
from pathlib import Path
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import LogUniform, Uniform, BanditPolicy
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType


ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), file_name=str(Path("..", "config.json"))
)


cluster = "gpu-cluster"

train_test_split = load_component(
    source=str(Path("..", "components", "train_test_split.yml"))
)
model_selection = load_component(
    source=str(Path("..", "components", "model_selection.yml"))
)
evaluate_model = load_component(
    source=str(Path("..", "components", "evaluate_model.yml"))
)


@pipeline(default_compute=cluster)
def tft_pipeline(
    dataset_input_path,
    train_size,
    num_folds,
):

    split_step = train_test_split(
        dataset_input_path=dataset_input_path,
        train_size=train_size,
        num_folds=num_folds,
    )

    model_selection_step = model_selection(
        train_input_path=split_step.outputs.train_output_path,
    )

    # # The result of the various runs is logged on the MLFlow server,
    # # but only output from the best run is returned by the step.
    # hyperparameter_search = model_selection_step.sweep(
    #     primary_metric="accuracy",
    #     goal="maximize",
    #     sampling_algorithm="bayesian",
    #     compute=cluster,
    # )

    # hyperparameter_search.set_limits(
    #     max_total_trials=20, max_concurrent_trials=2, timeout=36000
    # )
    
    # hyperparameter_search.early_termination = BanditPolicy(
    #     slack_factor= 0.1, delay_evaluation = 5, evaluation_interval = 1
    # )

    evaluate_step = evaluate_model(
        train_input_path=split_step.outputs.train_output_path,
        test_input_path=split_step.outputs.test_output_path,
        # The `outputs` attribute hyperparameter search contains the
        # output of the run that performed best
        model_input_path=model_selection_step.outputs.model_output_path,    # hyperparameter_search
    )

dataset = ml_client.data.get(name="raw_electricity_load", label="latest")

pipeline_job = tft_pipeline(
    dataset_input_path=Input(
        type=AssetTypes.URI_FILE,
        path=dataset.path,
    ),
    train_size=0.8,
    num_folds=3,
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="tft_experiment"
)



