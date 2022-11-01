"""utility.py: Utility code for Transformer fine-tuning on the IMDB Movie Reviews dataset."""

__author__ = "Oliver Atanaszov"
__email__ = "oliver.atanaszov@gmail.com"
__github__ = "https://github.com/ben0it8"
__copyright__ = "Copyright 2019, Planet Earth"

import sys
import os
import requests
import tarfile
from functools import wraps
from time import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
from datetime import timedelta

import re
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader

from pytorch_transformers import BertTokenizer, cached_path, AdamW

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler, PiecewiseLinear, create_lr_scheduler_with_warmup, ProgressBar

logger = logging.getLogger("utils.py")

num_cores = cpu_count()

# text and label column names
TEXT_COL = "text"
LABEL_COL = "label"

class TextProcessor:

    # special tokens for classification and padding
    CLS = '[CLS]'
    PAD = '[PAD]'

    def __init__(self, tokenizer, label2id: dict,
                 num_max_positions: int = 512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.num_max_positions = num_max_positions

    def process_example(self, example: Tuple[str, str]):
        "Convert text (example[0]) to sequence of IDs and label (example[1] to integer"
        assert len(example) == 2
        label, text = example[0], example[1]
        assert isinstance(text, str)
        tokens = self.tokenizer.tokenize(text)

        # truncate if too long
        if len(tokens) >= self.num_max_positions:
            tokens = tokens[:self.num_max_positions - 1]
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [
                self.tokenizer.vocab[self.CLS]
            ]
        # pad if too short
        else:
            pad = [self.tokenizer.vocab[self.PAD]
                   ] * (self.num_max_positions - len(tokens) - 1)
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [
                self.tokenizer.vocab[self.CLS]
            ] + pad

        return ids, self.label2id[label]


class Transformer(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"

    def __init__(self, embed_dim, hidden_dim, num_embeddings,
                 num_max_positions, num_heads, num_layers, dropout, causal):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(
        ), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(
                nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                              nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)),
                                   -float('Inf'),
                                   device=h.device,
                                   dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(
                self.layer_norms_1, self.attentions, self.layer_norms_2,
                self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h,
                             h,
                             h,
                             attn_mask=attn_mask,
                             need_weights=False,
                             key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithClfHead(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"

    def __init__(self, config, fine_tuning_config):
        super().__init__()
        self.config = fine_tuning_config
        self.transformer = Transformer(config.embed_dim,
                                       config.hidden_dim,
                                       config.num_embeddings,
                                       config.num_max_positions,
                                       config.num_heads,
                                       config.num_layers,
                                       fine_tuning_config.dropout,
                                       causal=not config.mlm)

        self.classification_head = nn.Linear(config.embed_dim,
                                             fine_tuning_config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        if isinstance(module,
                      (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, clf_tokens_mask, clf_labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)

        clf_tokens_states = (hidden_states *
                             clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(clf_logits.view(-1, clf_logits.size(-1)),
                            clf_labels.view(-1))
            return clf_logits, loss
        return clf_logits


def getenv_cast(var: str, cast=None):
    value = os.getenv(var)
    if value is not None and cast is not None:
        value = cast(value)
    return value


def freeze_body(model):
    for idx, (_, child) in enumerate(model.named_children()):
        if idx == 0:
            for param in child.parameters():
                param.requires_grad = False
            freeze_body(child)


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        delta = timedelta(seconds=round(time() - ts, 1))
        logger.info(f"Elapsed time for {f.__name__}: {str(delta):0>8}")
        return result

    return wrapper


def download_url(url: str,
                 dest: str,
                 overwrite: bool = True,
                 show_progress=True,
                 chunk_size=1024 * 1024,
                 timeout=4,
                 retries=5) -> None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    dest = os.path.join(dest, os.path.basename(url))
    if os.path.exists(dest) and not overwrite:
        logger.warning(f"File {dest} already exists!")
        return dest

    s = requests.Session()
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try:
        file_size = int(u.headers["Content-Length"])
    except:
        show_progress = False
    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress:
            pbar = tqdm(range(file_size),
                        leave=True,
                        desc=f"Downloading {os.path.basename(url)}")
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError:
            logger.warning(f"Download failed after {retries} retries.")
            import sys
            sys.exit(1)
        finally:
            return dest


@timeit
def untar(file_path, dest: str):
    "Untar `file_path` to `dest`"
    logger.info(f"Untar {os.path.basename(file_path)} to {dest}")
    with tarfile.open(file_path) as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path=str(dest))


def clean_html(raw: str):
    "remove html tags and whitespaces"
    cleanr = re.compile('<.*?>')
    clean = re.sub(cleanr, '  ', raw)
    return re.sub(' +', ' ', clean)


@timeit
def read_imdb(data_dir, max_lengths={"train": None, "test": None}):
    datasets = {}
    for t in ["train", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"imdb5k_{t}.csv"))
        maxlen = max_lengths.get(t)
        if maxlen is not None and maxlen <= len(df):
            df = df.sample(n=maxlen)
        df[TEXT_COL] = df[TEXT_COL].apply(lambda t: clean_html(t))
        datasets[t] = df
    return datasets


def process_row(processor, row):
    return processor.process_example((row[1][LABEL_COL], row[1][TEXT_COL]))


def create_dataloader(df: pd.DataFrame,
                      processor: TextProcessor,
                      batch_size: int = 32,
                      shuffle: bool = False,
                      valid_pct: float = None,
                      text_col: str = "text",
                      label_col: str = "label"):
    "Process rows in `df` with `processor` and return a  DataLoader"

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        result = list(
            tqdm(executor.map(process_row,
                              repeat(processor),
                              df.iterrows(),
                              chunksize=len(df) // 10),
                 desc=f"Processing {len(df)} examples on {num_cores} cores",
                 total=len(df)))

    features = [r[0] for r in result]
    labels = [r[1] for r in result]

    dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                            torch.tensor(labels, dtype=torch.long))

    if valid_pct is not None:
        valid_size = int(valid_pct * len(df))
        train_size = len(df) - valid_size
        valid_dataset, train_dataset = random_split(dataset,
                                                    [valid_size, train_size])
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        return train_loader, valid_loader

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=shuffle,
                             pin_memory=torch.cuda.is_available())
    return data_loader