import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import numpy as np
import ray
import typer
from numpyencoder import NumpyEncoder
from ray.air import Result
from ray.train.torch.torch_checkpoint import TorchCheckpoint
from typing_extensions import Annotated

from src.config import logger, mlflow
from src.data import CustomPreprocessor
from src.models import FinetunedLLM
from src.utils import collate_fn

app = typer.Typer()


def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d


class TorchPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch):
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self, batch):
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}

    def get_preprocessor(self):
        return self.preprocessor

    @classmethod
    def from_checkpoint(cls, checkpoint):
        metadata = checkpoint.get_metadata()
        preprocessor = CustomPreprocessor(class_to_index=metadata["class_to_index"])
        model = FinetunedLLM.load(Path(checkpoint.path, "args.json"), Path(checkpoint.path, "model.pt"))
        return cls(preprocessor=preprocessor, model=model)


def predict_proba(
    ds: ray.data.dataset.Dataset,
    predictor: TorchPredictor,
) -> List: 
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    outputs = preprocessed_ds.map_batches(predictor.predict_proba)
    y_prob = np.array([d["output"] for d in outputs.take_all()])
    results = []
    for i, prob in enumerate(y_prob):
        tag = preprocessor.index_to_class[prob.argmax()]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_class)})
    return results


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:  
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    run_id = sorted_runs.iloc[0].run_id
    print(run_id)
    return run_id


def get_best_checkpoint(run_id: str) -> TorchCheckpoint:  
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path  
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]


@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    title: Annotated[str, typer.Option(help="project title")] = None,
    description: Annotated[str, typer.Option(help="project description")] = None,
) -> Dict:  
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # Predict
    sample_ds = ray.data.from_items([{"title": title, "description": description, "tag": "other"}])
    results = predict_proba(ds=sample_ds, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__": 
    app()
