import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from datasets import load_dataset
from transformers import ChameleonProcessor
import pdb
import logging
from itertools import count

from teacher_cache import load_teacher_vector

logging.basicConfig(
    filename='qwenvl_sqa_4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S' 
)

#主函数，处理原始数据集，将文本tokenize，并处理图像数据。

def get_dataset(dataset, tokenizer, processor, max_size=1000000000, keep_image: bool = False):
    # 单个样本处理
    def tokenize_sample(sample, max_length=3400):
        image = sample["image"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        
        processed_question = sample["question"]

        # Tokenize question，tokenize后的input_id
        question_tokenized = sample["input_ids"]
        #logging.debug(f"step length: {len(sample["steps"])}")
        logging.debug(f"step length: {len(sample['steps'])}")
        # Tokenize steps,每一个step的内容都tokenize
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        sample["answer"] = str(sample["answer"])
        # Tokenize answer
        # 答案tokenize
        answer_tokenized = tokenizer.encode(
            "Therefore, the answer is " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        # Calculate total sequence length
        total_length = (
            len(question_tokenized)
            + sum(len(step) for step in steps_tokenized)
            + len(answer_tokenized)
        )
        print("question length: ", len(question_tokenized))
        # If total length exceeds max_length, truncate steps_tokenized
        # 计算应该保留的总token数（总长度减去超出的部分）
        if total_length > max_length:
            # Calculate how much to reduce
            excess_length = total_length - max_length
            # Reduce steps_tokenized
            new_steps_tokenized = []
            current_length = 0
            for step in steps_tokenized:
                if current_length + len(step) <= (sum(len(s) for s in steps_tokenized) - excess_length):
                    new_steps_tokenized.append(step)
                    current_length += len(step)
                else:
                    break
            steps_tokenized = new_steps_tokenized
        # Build the final sample
        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "idx": sample["idx"],
        }
        if keep_image:
            sample["image"] = image
        
        return sample

    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
    data = dataset
    # 功能：分布式环境下的处理
    # 只在主进程(rank=0)上处理数据
    # 使用32个进程并行处理
    # 将处理后的数据广播给其他进程
    # 移除原始特征列
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )

    return dataset


# 通过在左侧手动填充，确保一个 Batch 中所有样本的第一个 latent_id 都在同一个索引位置
# 保证了在 Batch 维度上，每一行的 Latent 标记在纵向上是完全对齐的
# 填充位的 position_id 被设为了 0。这意味着填充后的前两个 token（<pad> 和 A）的 position_id 都是 0
@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"
        #找到每个样本中第一个潜变量token的位置 
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]
        # 计算每个样本需要填充的数量，使潜变量位置对齐
        if len(earliest_latent) > 0:  
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                    #左侧pad位置，注意力掩码为0
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        # Tokenizer.pad can behave unexpectedly for extra tensor fields (e.g., pixel_values).
        # We only use it for input_ids/attention_mask, and stack the rest ourselves.
        to_pad = []
        extra_pixel_values = []
        extra_image_grid_thw = []
        extra_idx = []
        extra_images = []
        extra_teachers = {}

        for feature in features:
            to_pad.append(
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
            )
            extra_pixel_values.append(feature["pixel_values"])
            extra_image_grid_thw.append(feature["image_grid_thw"])
            extra_idx.append(feature.get("idx", -1))
            # For online SAM teacher: keep raw images as a python list.
            if "image" in feature:
                extra_images.append(feature["image"])

            for k, v in feature.items():
                if k.startswith("teacher_"):
                    extra_teachers.setdefault(k, []).append(v)

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            to_pad,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        batch["pixel_values"] = torch.stack([torch.as_tensor(x) for x in extra_pixel_values], dim=0)
        batch["image_grid_thw"] = torch.stack([torch.as_tensor(x) for x in extra_image_grid_thw], dim=0)
        batch["idx"] = torch.tensor(extra_idx, dtype=torch.long)
        if len(extra_images) > 0:
            batch["images"] = extra_images
        for k, vs in extra_teachers.items():
            batch[k] = torch.stack([torch.as_tensor(x).view(-1) for x in vs], dim=0)

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch

def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    expert_runtime=None,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # 正在训练的阶段
        scheduled_stage_to_train = scheduled_stage

        # 如果阶段超过最大限制
        # 跳过所有步骤（只输出答案）
        # 潜变量数量：根据配置决定是最大数量还是实际步骤数
        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )

        else:
            # 跳过几个stage，所以使用几个latent token
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        expert_segment = []
        if expert_runtime is not None and getattr(expert_runtime, "enabled", False):
            for e in expert_runtime.experts:
                expert_segment.extend(expert_runtime.prefix_ids[e])
                expert_segment.extend([expert_runtime.token_ids[e]] * int(expert_runtime.slots[e]))
                expert_segment.extend(expert_runtime.newline_ids)

        # 问题+latent token+除去skip阶段的cot过程
        tokens = (
            sample["question_tokenized"]
            + [latent_id] * n_latent_tokens
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + expert_segment
            + sample["answer_tokenized"]
        )

        out = {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
            )
            + tokens[
                n_latent_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "pixel_values": torch.tensor(sample["pixel_values"]),
            "image_grid_thw": sample["image_grid_thw"]
        }
        if "image" in sample:
            out["image"] = sample["image"]

        # Offline teacher vectors: one gt feature vector per expert per image.
        if (
            expert_runtime is not None
            and getattr(expert_runtime, "enabled", False)
            and expert_runtime.align_enabled
            and expert_runtime.align_weight > 0
            and expert_runtime.teacher_mode == "offline"
        ):
            if not expert_runtime.teacher_cache_dir:
                raise ValueError("expert.teacher.offline.cache_dir is required for offline mode")
            for e in expert_runtime.experts:
                vec = load_teacher_vector(expert_runtime.teacher_cache_dir, e, sample["idx"])
                if vec is None:
                    raise FileNotFoundError(
                        f"Missing teacher cache for expert={e}, idx={sample['idx']} under {expert_runtime.teacher_cache_dir}"
                    )
                out[f"teacher_{e}"] = vec

        # labels: 问题和潜变量部分设为-100（不计算损失），其余部分（剩余steps和answer）计算损失
        # attention_mask: 全部为1（都参与注意力计算）
        # 保留图像信息
        return out

    if torch.cuda.device_count() > 1:
        # 在0号卡处理后广播到其他device
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]

    else:
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
