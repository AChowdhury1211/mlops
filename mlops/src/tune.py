import datetime
import json
import os

import ray
import typer
from ray import tune
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from typing_extensions import Annotated

from src import data, train, utils
from src.config import EFS_DIR, MLFLOW_TRACKING_URI, logger

app = typer.Typer()


@app.command()
def tune_models(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    initial_params: Annotated[str, typer.Option(help="initial config for the tuning workload.")] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    num_runs: Annotated[int, typer.Option(help="number of runs in this tuning experiment.")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save results to.")] = None,
) -> ray.tune.result_grid.ResultGrid:
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )


    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config.get("num_samples", None))
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config, storage_path=EFS_DIR, local_dir=EFS_DIR)

    initial_params = json.loads(initial_params)
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2) 

    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],  
        grace_period=1,  
    )


    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )


    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )


    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=best_trial.metrics["trial_id"]),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:  
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":  
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    app()
