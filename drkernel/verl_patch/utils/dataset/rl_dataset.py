# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import copy
import json
import os
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.model import compute_position_id_with_mask

from kernel.rag.retriever import BM25ManualRetriever, format_retrieved_context


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO

    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key='prompt',
        image_key='images',
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,
        apply_chat_template=False,
        return_raw_chat=False,
        truncation='error',
        # if using sample_size, trucnate every validation dataset into this to reduce the time used for evaluation each time
        sample_size=None,
        filter_overlong_prompts=True,
        system_prompt_config=None,
        reference_template=None,
        rag_config=None,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.apply_chat_template = apply_chat_template
        self.truncation = truncation
        self.sample_size = sample_size
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt_config = system_prompt_config
        self.reference_template = reference_template
        self.rag_config = rag_config or {}
        self.rag_enabled = bool(self.rag_config.get("enable", False))
        self.rag_topk = int(self.rag_config.get("topk", 4))
        self.rag_max_ref_chars = int(self.rag_config.get("max_ref_chars", 4000))
        self.rag_query_mode = self.rag_config.get("query_mode", "op_signature")
        self.rag_max_chunks_per_source = self.rag_config.get("max_chunks_per_source", 1)
        self.rag_dedupe_query_tokens = bool(self.rag_config.get("dedupe_query_tokens", True))
        self.rag_fallback_to_plain = bool(self.rag_config.get("fallback_to_plain", True))
        self.rag_section_template = self.rag_config.get("section_template")
        self.rag_retriever = None
        if self.rag_enabled:
            kb_index_path = self.rag_config.get("kb_index_path")
            if not kb_index_path:
                raise ValueError("data.rag.kb_index_path must be set when RAG is enabled.")
            if not self.reference_template:
                raise ValueError("data.reference_template must be set when references are enabled.")
            if not self.rag_section_template:
                raise ValueError("data.rag.section_template must be set when RAG is enabled.")
            self.rag_retriever = BM25ManualRetriever(
                kb_index_path,
                query_mode=self.rag_query_mode,
                max_chunks_per_source=self.rag_max_chunks_per_source,
                dedupe_query_tokens=self.rag_dedupe_query_tokens,
            )

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _build_rag_section(self, prompt: str):
        if not self.rag_enabled or self.rag_retriever is None:
            return None

        chunks = self.rag_retriever.retrieve(prompt, topk=self.rag_topk)
        if not chunks and self.rag_fallback_to_plain:
            return None

        context = format_retrieved_context(chunks, max_chars=self.rag_max_ref_chars)
        if not context and self.rag_fallback_to_plain:
            return None

        return self.rag_section_template.format(context=context)

    def _normalize_chat(self, chat):
        if hasattr(chat, "tolist") and not isinstance(chat, list):
            chat = chat.tolist()
        return chat

    def _render_prompt(self, chat):
        if self.apply_chat_template:
            return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        if isinstance(chat, list):
            return chat[0]['content']
        return str(chat)

    def _replace_last_user_prompt(self, chat, prompt):
        if isinstance(chat, str):
            return prompt
        if not isinstance(chat, list):
            return chat

        updated_chat = copy.deepcopy(chat)
        for idx in range(len(updated_chat) - 1, -1, -1):
            message = updated_chat[idx]
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if isinstance(message.get("content", ""), str):
                updated_chat[idx]["content"] = prompt
            return updated_chat
        return updated_chat

    def _build_augmented_prompt_variants(self, prompt: str):
        rag_section = self._build_rag_section(prompt)

        variants = []
        if rag_section is not None:
            variants.append(
                self.reference_template.format(
                    prompt=prompt,
                    sections=rag_section,
                )
            )
        elif self.reference_template is not None:
            variants.append(
                self.reference_template.format(
                    prompt=prompt,
                    sections=""
                )
            )
        variants.append(prompt)

        deduped_variants = []
        seen = set()
        for variant in variants:
            if variant in seen:
                continue
            deduped_variants.append(variant)
            seen.add(variant)
        return deduped_variants

    def _maybe_apply_references(self, chat):
        # commit these two lines to open skill when rag is disabled
        if not self.rag_enabled:
            return chat, self._render_prompt(chat)

        if isinstance(chat, str):
            original_prompt = chat
        elif isinstance(chat, list):
            original_prompt = None
            for idx in range(len(chat) - 1, -1, -1):
                message = chat[idx]
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if isinstance(content, str):
                    original_prompt = content
                break
            if original_prompt is None:
                return chat, self._render_prompt(chat)
        else:
            return chat, self._render_prompt(chat)

        for candidate_prompt in self._build_augmented_prompt_variants(original_prompt):
            candidate_chat = self._replace_last_user_prompt(chat, candidate_prompt)
            rendered_prompt = self._render_prompt(candidate_chat)
            candidate_length = len(self.tokenizer.encode(rendered_prompt, add_special_tokens=False))
            if candidate_length <= self.max_prompt_length:
                return candidate_chat, rendered_prompt

        return chat, self._render_prompt(chat)

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            if self.sample_size is not None and len(dataframe) > self.sample_size:
                # use random state to ensure it can be reproducible
                dataframe = dataframe.sample(n=self.sample_size, random_state=42)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        if self.filter_overlong_prompts:
            # filter out too long prompts
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            if self.apply_chat_template:
                self.dataframe = self.dataframe[
                    self.dataframe.apply(
                        lambda doc: len(
                            tokenizer.encode(
                                tokenizer.apply_chat_template(
                                    doc[prompt_key], add_generation_prompt=True, tokenize=False
                                )
                            )
                        )
                        <= self.max_prompt_length,
                        axis=1,
                    )
                ]
            else:
                self.dataframe = self.dataframe[
                    self.dataframe.apply(
                        lambda doc: len(tokenizer.encode(doc[prompt_key][0]['content'])) <= self.max_prompt_length,
                        axis=1,
                    )
                ]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = self._normalize_chat(copy.deepcopy(row_dict.pop(self.prompt_key)))

        # Apply prompt from config file if provided
        if self.system_prompt_config is not None and self.apply_chat_template is True:
            # Load prompt config from file
            if os.path.exists(self.system_prompt_config):
                try:
                    with open(self.system_prompt_config) as f:
                        content = f.read().strip()

                    # Parse based on file extension
                    file_ext = os.path.splitext(self.system_prompt_config)[1].lower()
                    if file_ext == '.json':
                        config = json.loads(content)
                        prompt_text = config.get('prompt', '')
                        prompt_config_method = config.get('method', 'system')
                    elif file_ext in ['.yaml', '.yml']:
                        config = OmegaConf.load(self.system_prompt_config)
                        prompt_text = config.get('prompt', '')
                        prompt_config_method = config.get('method', 'system')
                    else:
                        # Treat as plain text
                        prompt_text = content
                        prompt_config_method = 'system'

                    # Apply prompt based on method
                    if prompt_config_method == 'system':
                        # Add system message to the beginning of the chat if not already present
                        if prompt_text and (not chat or chat[0].get('role') != 'system'):
                            chat = [{'role': 'system', 'content': prompt_text}] + chat
                    elif prompt_config_method == 'pre_input':
                        # Insert before the first user input
                        for i, message in enumerate(chat):
                            if message.get('role') == 'user':
                                new_content = prompt_text + message.get('content', '')
                                chat[i]['content'] = new_content
                                break
                    elif prompt_config_method == 'post_input':
                        # Insert after the first user input
                        for i, message in enumerate(chat):
                            if message.get('role') == 'user':
                                new_content = message.get('content', '') + prompt_text
                                chat[i]['content'] = new_content
                                break
                except Exception as e:
                    print(f"Error processing prompt config file: {e}")
            else:
                print(f"Prompt config file {self.system_prompt_config} does not exist.")
        elif self.system_prompt_config is not None and self.apply_chat_template is False:
            raise ValueError("Error: system_prompt_config is provided but apply_chat_template is False.")

        chat, prompt_with_chat_template = self._maybe_apply_references(chat)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>'
                        + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length)
                        + '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<|placeholder|>', self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist() if not isinstance(chat, list) else chat

        # add index for each prompt
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        index = extra_info.get("index", 0)
        row_dict["index"] = index
        row_dict["prompt_index"] = item

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


class SolveRateDynamicRLHFDataset(RLHFDataset):
    """
    RLHF Dataset with dynamic solve rate tracking capabilities

    Attributes:
        current_solve_rates (np.ndarray): Array of current solve rates for samples
        original_indices (list): Preserved original indices for data access

    Args:
        Inherits all arguments from RLHFDataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate dataset structure
        if 'solve_rate' not in self.dataframe.columns:
            raise ValueError("Dataset must contain 'solve_rate' column")

        # Initialize solve rate tracking
        self.current_solve_rates = self.dataframe['solve_rate'].to_numpy().copy()
        # self.original_indices = self.dataframe.index.tolist()

    def __len__(self):
        return super().__len__()
