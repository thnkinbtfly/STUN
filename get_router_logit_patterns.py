#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import collections
import copy
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, HqqConfig
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from torch.utils.data import Dataset as torchDataset


IGNORE_TOKEN_ID = -100

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.36.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from typing import Optional, List


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        # default=None,
        default='allenai/c4',
        # default='vicgalle/alpaca-gpt4',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument('--train_dir', default=None, type=str)
    parser.add_argument('--torch_dtype', default=None)
    parser.add_argument(
        "--train_file", type=str,
        default=None,
        help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument('--method', default='')
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=0,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='Mixtral-8x7B-Instruct-v0.1',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--output_dir", type=str,
                        default='router_logits_mixtral',
                        help="Where to store the final model.")
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument('--pad', default='')
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument('--optim', default='adamw_torch')
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--num_warmup_ratio", type=float, default=0.03, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_samples", type=int, default=1000, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=64,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument('--attn_implementation', type=str, default='sdpa')
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--intermediate_id", type=int, default=23)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--loss_w", type=float, default=1)
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument('--convert_state_dict', action='store_true', default=False)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None and args.train_dir is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension == 'jsonl':
                extension = 'json'
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if 'c4' in args.dataset_name:
            raw_datasets = load_dataset(
                'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
            )
        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            raw_datasets = raw_datasets['train']
    else:
        if args.train_dir is not None:
            import glob
            raw_datasets = load_dataset(
                'json',
                data_files=glob.glob(os.path.join(args.train_dir, '*.jsonl')),
            )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = 'json'
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        # quant_config = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False,
        #                          axis=0)  # axis=0 is used by default
        # device_map = {
        #     'model.embed_tokens': 0,
        #     'model.norm': 7,
        #     'lm_head': 7,
        # }
        # for l in range(35):
        #     device_map[f'model.layers.{l}'] = (l + 1) // 5

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            # low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            # quantization_config=quant_config,
            attn_implementation="sdpa"
            # attn_implementation="flash_attention_2"
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if args.pad == 'pad':
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets.column_names
    text_column_name = 'text'

    use_eos = True
    eos = tokenizer.eos_token if use_eos else ""
    # input_mask = False if args.sft_stage == 1 else True

    def tokenize_alpaca(example):
        def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
            """Tokenize a list of strings."""
            tokenized = tokenizer(
                strings,
                # return_tensors="pt",
                # max_length=tokenizer.model_max_length,
                # truncation=True,
            )
            input_ids = tokenized.input_ids[0]
            return input_ids
            # return dict(
            #     input_ids=input_ids,
            # )

        def preprocess(
                sources,
                targets,
                tokenizer: transformers.PreTrainedTokenizer,
        ):
            """Preprocess the data by tokenizing."""
            import copy
            IGNORE_INDEX = -100
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
            input_ids = examples_tokenized
            labels = copy.deepcopy(input_ids)
            labels = [IGNORE_INDEX] * len(sources_tokenized) + labels[len(sources_tokenized):]
            # labels[:len(sources_tokenized)] = IGNORE_INDEX
            return dict(input_ids=input_ids, labels=labels)

        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        prompt_dict = PROMPT_DICT
        example = {k: v for k,v in example.items()}

        prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}"]
        data_dict = preprocess(sources, targets, tokenizer)

        # en_targets = [f"{example['en_output']}{tokenizer.eos_token}"] # consider the parallel en_output for intermediate gen
        # en_data_dict = preprocess(sources, en_targets, tokenizer)
        # for k, v in en_data_dict.items():
        #     data_dict['en_' + k] = v

        # if args.enin_also:
        #     example['instruction'] = example['en_instruction']
        #     example['input'] = example['en_input']
        #     prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
        #     en_sources = [
        #         prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     ]
        #     en_data_dict = preprocess(en_sources, targets, tokenizer)
        #     for k, v in en_data_dict.items():
        #         data_dict['enin_' + k] = v
        #
        #     en_data_dict = preprocess(en_sources, en_targets, tokenizer)
        #     for k, v in en_data_dict.items():
        #         data_dict['enin_en_' + k] = v

        return data_dict

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, 2048)} instead. You can change that default value by passing --block_size xxx."
            )
            if 2048 > 0:
                block_size = min(1024, 2048)
            else:
                block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)


    if args.train_file and 'openhermes' in args.train_file:
        def preprocess(
                sources,
                tokenizer: transformers.PreTrainedTokenizer,
        ) -> Dict:
            conv = get_conversation_template("vicuna_v1.1")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            # Apply prompt templates
            conversations = []
            for i, source in enumerate(sources):
                if source[0]["from"] == "system":
                    conv.set_system_message(source[0]["value"])
                    source.pop(0)

                if roles[source[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]

                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    assert role == conv.roles[j % 2], f"{i}"
                    conv.append_message(role, sentence["value"])
                conversations.append(conv.get_prompt())

            from datasets import Dataset
            dataset = Dataset.from_dict({"convs": conversations})
            # print(len(dataset))
            for ex in dataset:
                # print(ex)
                break

            def tok(conversation):
                conversation = conversation['convs']
                # Tokenize conversations
                input_ids = tokenizer(
                    conversation,
                    return_tensors="pt",
                    # padding="max_length",
                    max_length=block_size,
                    truncation=True,
                ).input_ids[0]
                target = input_ids.clone()

                assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

                # Mask targets. Only compute loss on the assistant outputs.
                sep = conv.sep + conv.roles[1] + ": "
                # total_len = int(target.ne(tokenizer.pad_token_id).sum())

                turns = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, turn in enumerate(turns):
                    if turn == "":
                        break
                    turn_len = len(tokenizer(turn).input_ids)

                    parts = turn.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        instruction_len -= 1

                    # Ignore the user instructions
                    target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        cur_len -= 1

                target[cur_len:] = IGNORE_TOKEN_ID

                # if False:  # Inspect and check the correctness of masking
                #     z = target.clone()
                #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                #     exit()
                #
                # if cur_len < tokenizer.model_max_length:
                #     if cur_len != total_len:
                #         target[:] = IGNORE_TOKEN_ID

                return dict(
                    input_ids=input_ids,
                    labels=target,
                    attention_mask=torch.ones_like(input_ids),
                )

            dataset = dataset.map(tok, num_proc=64, remove_columns='convs')
            return dataset

        class SupervisedDataset(torchDataset):
            """Dataset for supervised fine-tuning."""

            def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
                super(SupervisedDataset, self).__init__()

                sources = [example["conversations"] for example in raw_data]
                random.seed(42)
                random.shuffle(sources)
                sources = sources[:1000]
                self.data_dict = preprocess(sources, tokenizer)
                # data_dict = preprocess(sources, tokenizer).to_dict()
                # self.input_ids = data_dict["input_ids"]
                # self.labels = data_dict["labels"]
                # self.attention_mask = data_dict["attention_mask"]

            def __len__(self):
                return len(self.data_dict)
                # return len(self.input_ids)

            def __getitem__(self, i) -> Dict[str, torch.Tensor]:
                return self.data_dict[i]
                # return dict(
                #     input_ids=self.input_ids[i],
                #     labels=self.labels[i],
                #     attention_mask=self.attention_mask[i],
                # )

        train_dataset = SupervisedDataset(json.load(open(args.train_file, "r")), tokenizer)

    else:
        def tokenize_function(examples):
            output = tokenizer(examples[text_column_name])
            return output

        raw_datasets = raw_datasets.shuffle(seed=42).select(
            range(args.num_samples * 16))

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            train_dataset = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

        train_dataset = train_dataset.select(range(args.num_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")



    # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )
    train_dataloader = DataLoader(
        train_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        shuffle=False
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim_kwargs = {}
    if args.optim == 'adamw_torch':
        optimizer_cls = torch.optim.AdamW
    else:
        raise NotImplementedError


    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate, **optim_kwargs)

    # Scheduler and math around the number of training steps.
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    num_warmup_steps = args.num_warmup_steps * args.gradient_accumulation_steps
    if args.num_warmup_ratio:
        num_warmup_steps = round(args.num_warmup_ratio * args.max_train_steps)
    print(num_warmup_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)


    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()


    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)




    with torch.no_grad():
        no_output_router_logits = False
        model.eval()
        os.makedirs(args.output_dir, exist_ok=True)
        for step, batch in enumerate(train_dataloader):
            progress_bar.update(1)
            if step >= 1000 / args.per_device_eval_batch_size:
                break
            if not no_output_router_logits:
                try:
                    outputs = model(**batch, output_router_logits=True, return_dict=True)
                except:
                    no_output_router_logits = True
                    outputs = model(**batch, return_dict=True)
            else:
                outputs = model(**batch, return_dict=True)
            router_logits = outputs.router_logits
            print(len(router_logits))
            print(router_logits[0].shape)
            torch.save(router_logits, os.path.join(args.output_dir, f"router_logits_{step:04d}.pt"))

if __name__ == "__main__":
    main()
