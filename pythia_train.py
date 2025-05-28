#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import re

import datasets
from evaluate import load
import torch
from torch.optim import AdamW
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    is_torch_xla_available,
    set_seed,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    TrainerCallback
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.47.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    revisions: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of revisions. "
                "Provide a list of pt steps."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_input_length: int = field(default=256, metadata={"help": "Maximum length for the input sequence."})
    max_output_length: int = field(default=4, metadata={"help": "Maximum length for the output sequence."}) # For [yes, no]
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`test_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_revision != "main":
        training_args.output_dir += f"/pretrain_checkpoints/{model_args.model_revision}/"
        training_args.save_steps = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Check if we need to overwrite the output directory and clean existing prediction files
    if training_args.overwrite_output_dir:
        eval_prediction_file = os.path.join(training_args.output_dir, "eval_predictions.txt")
        test_prediction_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        # Clear existing prediction files if they exist
        if os.path.exists(eval_prediction_file):
            open(eval_prediction_file, "w").close()
        if os.path.exists(test_prediction_file):
            open(test_prediction_file, "w").close()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    # WE ADD HERE A SPECIAL TOKEN TO THE MODEL VOCAB: 
    special_token = "[N_GRAM]"
    tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    if data_args.source_column is None:
        source_column = column_names[0]
        logger.warning("No explicit source column is given. Be careful")
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = column_names[1]
        logger.warning("No explicit target column is given. Be careful")
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_function(examples):
        sources = examples[source_column]
        targets = examples[target_column]
        
        input_ids_list = []
        labels_list = []
        
        for source, target in zip(sources, targets):
            input_encodings = tokenizer(source, max_length=data_args.max_input_length, truncation=True, padding=False)
            
            target_encodings = tokenizer(target, max_length=data_args.max_output_length, truncation=True, padding=False)
            
            input_ids = input_encodings["input_ids"] + target_encodings["input_ids"]
            labels = [-100] * len(input_encodings["input_ids"]) + target_encodings["input_ids"]
            
            padding_length = data_args.max_input_length + data_args.max_output_length
            input_ids = input_ids + [tokenizer.pad_token_id] * (padding_length - len(input_ids))
            labels = labels + [-100] * (padding_length - len(labels))
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # logger.info(f"Input_ids example {input_ids_list[10]}")
        # logger.info(f"Labels example {labels_list[10]}")
        
        return {"input_ids": input_ids_list, "labels": labels_list}
    
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            print(train_dataset["input_ids"][0])
            print(train_dataset["labels"][0])

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[x for x in column_names if x != "source_text"], # we keep the first column to log the source text
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    print("COLUMN names:", column_names)

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[x for x in column_names if x != "source_text"], # we keep the first column to log the source text
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    

    if hasattr(config, "max_position_embeddings"):
        max_position_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_position_embeddings = 1024

    # Check if the total length exceeds max_position_embeddings
    total_length = data_args.max_input_length + data_args.max_output_length
    if total_length > max_position_embeddings:
        raise ValueError(
            f"The combined length of max_input_length ({data_args.max_input_length}) and max_output_length "
            f"({data_args.max_output_length}) is {total_length}, which exceeds the model's max_position_embeddings "
            f"({max_position_embeddings}). Consider reducing the input/output lengths or increasing "
            "max_position_embeddings if modifying the model architecture."
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    accuracy_metric = load("accuracy", cache_dir=model_args.cache_dir, average = "macro")
    precision_metric = load("precision", cache_dir=model_args.cache_dir, average = "macro")
    recall_metric = load("recall", cache_dir=model_args.cache_dir, average = "macro")

    do_test = False # flag to check if we are in test mode

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]

        # print(labels[0])

        target_indices = [np.where(seq != -100)[0][-1] for seq in labels]
        # print(target_indices[0])

        # Replace -100s used for loss as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_target = [seq[idx -1] == 9820 for seq, idx in zip(preds, target_indices)] # - 1 due to autoregressiveness (9820 = yes token)
        labels_target = [seq[idx] == 9820 for seq, idx in zip(labels, target_indices)]

        # Retrieve original inputs for each prediction from eval_dataset
        original_inputs = [example[data_args.source_column] for example in eval_dataset] if not do_test else [example[data_args.source_column] for example in predict_dataset]

        print(preds[0], decoded_preds[0], preds_target[0])  # Example output
        print(labels[0], decoded_labels[0], labels_target[0])  # Example output

        eval_prediction_file = os.path.join(training_args.output_dir, "eval_predictions.txt")
        classes_preds = {str(k):{"preds": [], "labels": []} for k in range(1, 7)}
        classes_preds.update({f"_{k}":{"preds": [], "labels": []} for k in range(1, 7)})
        classes_preds.update({f"{k}_":{"preds": [], "labels": []} for k in range(1, 7)})
        with open(eval_prediction_file, "a") as writer:
            for i, pred in enumerate(preds_target):
                # Retrieve the original input text for the i-th prediction
                original_input = original_inputs[i]
                decoded_pred = pred
                full_decoded_pred = decoded_preds[i]
                label = labels_target[i]
                # n_gram = re.findall("^(.*?) inside", original_input)[0]
                n_gram = re.findall("Is \[N_GRAM\](.*?)\[N_GRAM\] inside", original_input)[0]
                if n_gram.startswith("_"):
                    classes_preds[f"_{len(n_gram)-1}"]["preds"].append(pred)
                    classes_preds[f"_{len(n_gram)-1}"]["labels"].append(label)
                elif n_gram.endswith("_"):
                    classes_preds[f"{len(n_gram)-1}_"]["preds"].append(pred)
                    classes_preds[f"{len(n_gram)-1}_"]["labels"].append(label)
                else:
                    classes_preds[f"{len(n_gram)}"]["preds"].append(pred)
                    classes_preds[f"{len(n_gram)}"]["labels"].append(label)

                if not do_test:
                    writer.write(f"Input: {original_input} --- Prediction: {decoded_pred} --- Label: {label} --- Full prediction: {full_decoded_pred}\n")

        accuracy = accuracy_metric.compute(predictions=preds_target, references=labels_target)
        precision = precision_metric.compute(predictions=preds_target, references=labels_target, average = "macro")
        recall = recall_metric.compute(predictions=preds_target, references=labels_target, average = "macro")

        results = {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
        }

        for n_gram_class in classes_preds:
            preds = classes_preds[n_gram_class]["preds"]
            labels = classes_preds[n_gram_class]["labels"]
            accuracy = accuracy_metric.compute(predictions=preds, references=labels)
            precision = precision_metric.compute(predictions=preds, references=labels, average = "macro")
            recall = recall_metric.compute(predictions=preds, references=labels, average = "macro")
            results[f"-{n_gram_class}-accuracy"] = accuracy["accuracy"]
            results[f"-{n_gram_class}-precision"] = precision["precision"]
            results[f"-{n_gram_class}-recall"] = recall["recall"]

        return results

    optimizer = AdamW(model.parameters(), lr = training_args.learning_rate, weight_decay = training_args.weight_decay)
    if training_args.do_train:
        n_steps = int(train_dataset.num_rows / (training_args.per_device_train_batch_size * (torch.cuda.device_count()) * training_args.gradient_accumulation_steps) * training_args.num_train_epochs)
    else:
        n_steps = 0
    warmup_steps = int(n_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=n_steps)

    # class GradientLoggingCallback(TrainerCallback):
    #     def on_step_end(self, args, state, control, **kwargs):
    #         if (state.global_step + 1) % args.gradient_accumulation_steps == 0:
    #         # Get the model from the arguments
    #             model = kwargs['model']
    #             grad_info = {"layers": [], "ave_grads": [], "max_grads": []}
                
    #             # Only proceed if model parameters have gradients
    #             for name, param in model.named_parameters():
    #                 if param.requires_grad and param.grad is not None:  # Check both requires_grad and actual gradient presence
    #                     grad_info["layers"].append(name)
    #                     grad_info["ave_grads"].append(param.grad.abs().mean().item())
    #                     grad_info["max_grads"].append(param.grad.abs().max().item())
                
    #             # If gradients are missing, this log will indicate it
    #             if not grad_info["layers"]:
    #                 logger.warning("No gradients were recorded for this step.")

    #             # Log gradient norms
    #             for layer, ave_grad, max_grad in zip(grad_info["layers"], grad_info["ave_grads"], grad_info["max_grads"]):
    #                 logger.info(f"Layer: {layer}, Avg Grad: {ave_grad}, Max Grad: {max_grad}")


    def custom_collator(features):
        batch = {'input_ids': [], 'labels': []}
        for feature in features:
            batch['input_ids'].append(feature['input_ids'])
            batch['labels'].append(feature['labels'])
        
        batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long)
        batch['labels'] = torch.tensor(batch['labels'], dtype=torch.long)
        return batch



    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers = (optimizer, scheduler),
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=custom_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        # callbacks=[GradientLoggingCallback()]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # test
    if training_args.do_predict:
        do_test = True
        logger.info("*** Predict ***")
        
        predict_results = trainer.predict(metric_key_prefix="predict", test_dataset=predict_dataset)

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        preds, labels = predict_results.predictions, predict_results.label_ids

        target_indices = [np.where(seq != -100)[0][-1] for seq in labels]

        # Replace -100s used for loss as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_target = [seq[idx -1] == 9820 for seq, idx in zip(preds, target_indices)] # - 1 due to autoregressiveness (9820 = yes token)
        labels_target = [seq[idx] == 9820 for seq, idx in zip(labels, target_indices)]

        # Retrieve original inputs for each prediction from eval_dataset
        original_inputs = [example[data_args.source_column] for example in predict_dataset]

        print(preds[0], decoded_preds[0], preds_target[0])  # Example output
        print(labels[0], decoded_labels[0], labels_target[0])  # Example output

        prediction_file = os.path.join(training_args.output_dir, "test_predictions.txt")

        with open(prediction_file, "a") as writer:
            for i, pred in enumerate(preds_target):
                # Retrieve the original input text for the i-th prediction
                original_input = original_inputs[i]
                decoded_pred = pred
                full_decoded_pred = decoded_preds[i]
                label = labels_target[i]
                writer.write(f"Input: {original_input} --- Prediction: {decoded_pred} --- Label: {label} --- Full prediction: {full_decoded_pred}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # add eventual code for running inference on the test dataset


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()