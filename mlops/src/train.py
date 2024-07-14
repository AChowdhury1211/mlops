import datetime
import json
import os
import tempfile
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertModel
from typing_extensions import Annotated

from src import data, utils
from src.config import EFS_DIR, MLFLOW_TRACKING_URI, logger
from src.models import FinetunedLLM

app = typer.Typer()


def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:  
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad() 
        z = model(batch) 
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  
        J = loss_fn(z, targets)  
        J.backward()  
        optimizer.step() 
        loss += (J.detach().item() - loss) / (i + 1)  
    return loss


def eval_step(
    ds: Dataset, batch_size: int, model: nn.Module, num_classes: int, loss_fn: torch.nn.modules.loss._WeightedLoss
) -> Tuple[float, np.array, np.array]: 
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None: 
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    utils.set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")


    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    model = train.torch.prepare_model(model)


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers
    for epoch in range(num_epochs):

        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        with tempfile.TemporaryDirectory() as dp:
            if isinstance(model, DistributedDataParallel):  
                model.module.save(dp=dp)
            else:
                model.save(dp=dp)
            metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
            checkpoint = Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    train_loop_config: Annotated[str, typer.Option(help="arguments to use for training.")] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save results to.")] = None,
) -> ray.air.result.Result:
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
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


    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    results = trainer.fit()
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(experiment_name=experiment_name, trial_id=results.metrics["trial_id"]),
        "params": results.config["train_loop_config"],
        "metrics": utils.dict_to_list(results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]),
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
