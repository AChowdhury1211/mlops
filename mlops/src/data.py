import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src.config import STOPWORDS


def load_data(dataset_loc: str, num_samples: int = None) -> Dataset:
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:  
        return df[df["_split"] == split].drop("_split", axis=1)

  
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas") 
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")  
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas") 

    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    text = text.lower()

    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text) 
    text = re.sub("[^A-Za-z0-9]+", " ", text)  
    text = re.sub(" +", " ", text)  
    text = text.strip()  
    text = re.sub(r"http\S+", "", text)  

    return text


def tokenize(batch: Dict) -> Dict:
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))


def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    df["text"] = df.title + " " + df.description 
    df["text"] = df.text.apply(clean_text)  
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  
    df = df[["text", "tag"]]  
    df["tag"] = df["tag"].map(class_to_index)  
    outputs = tokenize(df)
    return outputs


class CustomPreprocessor:
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, ds):
        return ds.map_batches(preprocess, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")
