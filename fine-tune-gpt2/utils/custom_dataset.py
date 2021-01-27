import json
import os
import pickle
import random
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

# from ...utils import logging


# logger = logging.get_logger(__name__)


class CustomDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer,
        file_path: str,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
        use_subset: float = 1.0,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}".format(
                tokenizer.__class__.__name__,
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                # logger.info(
                #     f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                # )

            else:
                # logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                text_entries = text.split('<BOS>')[1:]

                for i in range(0, len(text_entries)):
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_entries[i]))
                    if len(tokenized_text) > 1024:
                        continue
                    tokenized_text += [tokenizer.pad_token_id] * (1024 - len(tokenized_text))
                    self.examples.append(tokenized_text)
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # logger.info(
                #     "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                # )
            
            # use only a subset of the data
            self.examples = self.examples[:int(len(self.examples) * use_subset)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

