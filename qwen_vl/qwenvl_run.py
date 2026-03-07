# import torch
# from torch_npu.contrib import transfer_to_npu
# # import os
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import torch.distributed
# import torch.optim as optim
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datetime import timedelta
# import deepspeed
# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# from torch.optim import AdamW
# import shutil
# import numpy as np
# from torch.utils.data import Subset
# from collections import OrderedDict
# import re
# import wandb

# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from transformers.models.gpt2.modeling_gpt2 import GPT2Block
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# import logging
# logging.basicConfig(
#     filename='qwenvl4.log',  
#     level=logging.DEBUG,          
#     format='[%(asctime)s] %(message)s',  
#     datefmt='%Y-%m-%d %H:%M:%S' 
# )

# from qwen_ivtlr import IVTLR

# from tqdm import tqdm
# from copy import copy
# import itertools
# import os, sys
# import yaml
# import json
# import gc
# import argparse
# import functools
# from utils import Config, set_seed
# import pdb
# from peft import LoraConfig, get_peft_model
# from dataset import (
#     get_dataset,
#     get_cot_latent_dataset,
#     MyCollator,
# )

# # LoRA
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     r=64,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     inference_mode=False
# )














# def main():
#     print("Initializing DeepSpeed Training!")
#     parser = argparse.ArgumentParser(description="ivtlr")
#     parser.add_argument("config_file")
#     parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
#     parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
#     parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
#     args = parser.parse_args()

#     # Initialize DeepSpeed
#     deepspeed.init_distributed()
#     local_rank = args.local_rank
#     rank = int(os.environ['RANK'])
#     world_size = int(os.environ['WORLD_SIZE'])
#     torch.cuda.set_device(local_rank)
#     print("line 57")
#     # load the configuration file
#     with open(args.config_file) as f:
#         config_dict = yaml.safe_load(f)

#     configs = Config(config_dict)
#     set_seed(configs.seed)
#     save_dir = os.path.join(configs.save_path, configs.name)

#     if not os.path.exists(save_dir) and rank == 0:
#         os.makedirs(save_dir)

#     torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

#     cur_ckpts = os.listdir(save_dir)


#     # check if the job is preempted and resumed.
#     if len(cur_ckpts) > 0 and rank == 0:
#         raise ValueError(
#             f"Save directory {save_dir} is not empty! "
#         )

#     if configs.resume != 0:
#         # by setting `resume`, we can skip a few epoches at the beginning.
#         print(
#             f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
#         )
        
        
        
#     print("start loading model")
#     # Todo:modify model and Tokenizer
#     # model = Qwen2VLForConditionalGeneration.from_pretrained(
#     #     "Qwen/Qwen2-VL-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
#     # )
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
# )
#     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
#     #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=False, trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained("/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", use_fast=False, trust_remote_code=True)
#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.add_tokens("<|start-latent|>")
#     tokenizer.add_tokens("<|end-latent|>")
#     tokenizer.add_tokens("<|latent|>")
#     # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", tokenizer=tokenizer)
#     processor = AutoProcessor.from_pretrained("/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", tokenizer=tokenizer)
#     latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
#     print("latent_id: ", latent_id)
#     start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
#     end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
#     image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
#     visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
#     visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

#     model = get_peft_model(model, lora_config)

#     loaded = False

#     model.resize_token_embeddings(len(tokenizer))
#     embeddings = model.get_input_embeddings()
#     target_id = tokenizer.convert_tokens_to_ids("<<")
#     # initialize the new token embeddings with a known token
#     # it helps stablize the training
#     for token_id in [latent_id, start_id, end_id]:
#         target_embedding = embeddings.weight.data[token_id]
#         embeddings.weight.data[token_id] = target_embedding
#         # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
#         lm_head = model.lm_head
#         lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
#     model.print_trainable_parameters()

#     model = IVTLR(model, latent_id, start_id, end_id, tokenizer.eos_token_id, image_token_id, visual_start_id, visual_end_id)

#     print(f"Running Deepspeed on rank = {rank}, world size = {world_size}")
#     model = model.to(rank)
    
#     if configs.bf16:
#         model.to(torch.bfloat16)

#     model_engine, optimizer, _, _ = deepspeed.initialize(
#         model=model,
#         config=args.deepspeed_config,
#         # optimizer = optimizer,
#         model_parameters=filter(lambda p: p.requires_grad, model.parameters())
#     )

#     del model

#     dataset = load_dataset("/home/ma-user/work/lbx/IVT-LR/data/M3CoT_data")

#     def process_example(example):
#         rationale = example["rationale"].replace("\n", " ").strip()
#         example["steps"] = rationale.split(". ")
#         if example["steps"][-1] == "":
#             example["steps"].pop()

#         if len(example["steps"]) > 3:
#             total_steps = len(example["steps"])
#             step_size = total_steps // 3
#             remainder = total_steps % 3

#             new_steps = []
#             start = 0

#             for i in range(3):
#                 end = start + step_size + (1 if i < remainder else 0)
#                 new_steps.append(". ".join(example["steps"][start:end]))
#                 start = end

#             example["steps"] = new_steps


#         question = example["question"]
#         choices = example["choices"]
        
#         #两个大括号 {{ ：表示一个普通的大括号字符 { 在 f-string 中，单括号 { 有特殊含义（用于插入变量）。
#         #如果你想在最终生成的字符串里显示一个真正的 {，你就必须连续写两次进行“转义”。
#         choices_str = "[Options]:\n"+"\n".join([
#             f"({chr(65 + i)}).{{{choice.strip()}}}"
#             for i, choice in enumerate(choices)
#         ])
#         question = question
#         question_with_braces = f"{{{question.strip()}}}"
#         prefix_str = "Answer:"
        
#         example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"
        
#         del example["rationale"]
#         del example["choices"]

#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
#                 {"type": "text", "text": example["question"]}
#             ]
#         }]

#         example["question"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[example["question"]],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         )
#         inputs = {k: v.tolist() for k, v in inputs.items()}
#         example["input_ids"] = torch.tensor(inputs["input_ids"][0])
#         example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
#         example["pixel_values"] = torch.tensor(inputs["pixel_values"])

#         return example
    
#     print("start dataset")


#     # dataset = load_dataset("derek-thomas/ScienceQA")

#     # def process_example_sqa(example):
#     #     example["answer"] = str(example["answer"])
#     #     # rationale：merge lecture and solution
#     #     lecture = example.get("lecture", "") or ""
#     #     solution = example.get("solution", "") or ""
        
#     #     # merge lecture and solution
#     #     if lecture and solution:
#     #         rationale = (lecture.strip() + " " + solution.strip()).strip()
#     #     elif lecture:
#     #         rationale = lecture.strip()
#     #     elif solution:
#     #         rationale = solution.strip()
#     #     else:
#     #         # both is null
#     #         rationale = example["answer"]
#     #         print(f"Warning: Both lecture and solution are empty for question: {example['question']}")
#     #         rationale = str(rationale)

#     #     #移除末尾空值
#     #     rationale = rationale.replace("\n", " ").strip()
#     #     example["steps"] = rationale.split(". ")
#     #     if example["steps"][-1] == "":
#     #         example["steps"].pop()

#     #     # multi-steps，各个steps合并（均匀）
#     #     if len(example["steps"]) > 3:
#     #         total_steps = len(example["steps"])
            
#     #         step_size = total_steps // 3
#     #         remainder = total_steps % 3

            
#     #         new_steps = []
#     #         start = 0
           
#     #         for i in range(3):
#     #             end = start + step_size + (1 if i < remainder else 0)
#     #             new_steps.append(". ".join(example["steps"][start:end]))
#     #             start = end

#     #         example["steps"] = new_steps
        
#     #     question = example["question"]
#     #     choices = example["choices"]
#     #     choices_str = "[Options]:\n"+"\n".join([
#     #         f"({chr(65 + i)}).{{{choice.strip()}}}"  
#     #         for i, choice in enumerate(choices)
#     #     ])

#     #     question = question
#     #     question_with_braces = f"{{{question.strip()}}}"
#     #     prefix_str = "Answer:"
        
#     #     example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"

        
#     #     # 4. 构建 Qwen2-VL 的输入消息
#     #     # 注意：ScienceQA 的 image 可能是 PIL 对象，process_vision_info 和 processor 可以直接处理
#     #     messages = [{
#     #         "role": "user",
#     #         "content": [
#     #             # 保持与 M3CoT 一致的 resize 策略，或者根据显存情况去掉 resize
#     #             {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
#     #             {"type": "text", "text": example["question"]}
#     #         ]
#     #     }]
        
#     #     prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     #     image_inputs, video_inputs = process_vision_info(messages)
#     #     inputs = processor(
#     #         text=[prompt_text],
#     #         images=image_inputs,
#     #         videos=video_inputs,
#     #         padding=True,
#     #         return_tensors="pt"
#     #     )
        
#     #     inputs = {k: v.tolist() for k, v in inputs.items()}
#     #     example["input_ids"] = torch.tensor(inputs["input_ids"][0])
#     #     example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
#     #     example["pixel_values"] = torch.tensor(inputs["pixel_values"])
        
#     #     # 删除不需要的列
#     #     if "lecture" in example:
#     #         del example["lecture"]
#     #     if "solution" in example:
#     #         del example["solution"]
#     #     if "choices" in example:
#     #         del example["choices"]

#     #         # 确保 answer 是字符串，供 dataset.py 使用
#     #     example["answer"] = str(example["answer"]) # SQA answer 是 int (索引)，需要转 string 吗？
#     #     # 注意：dataset.py 中 tokenize_sample 会用到 "Therefore, the answer is " + sample["answer"]
#     #     # 对于 SQA，answer 通常是 0, 1, 2... 对应的 A, B, C?
#     #     # Chameleon 代码中做了转换： sample["answer"] = str(sample["answer"]) 
#     #     # 并且 extract answer 时是找 0, 1 或 A, B. 
#     #     # 这里建议将 int index 转为 A/B/C 可能会更好，或者保持 int string。
#     #     # 考虑到 dataset.py 代码： tokenizer.encode("Therefore, the answer is " + sample["answer"])
#     #     # 如果 sample["answer"] 是 '0'，模型学到的就是 "...is 0"。如果是 'A'，就是 "...is A"。
#     #     # 建议转为选项字母以保持与 Options 一致：
#     #     if isinstance(example["answer"], int):
#     #         example["answer"] = chr(65 + example["answer"]) # 0 -> A, 1 -> B
            
#     #     return example

        
#     # print("start dataset")
#     # # 处理 train 数据集

    

#     def has_image(example):
#         return (
#             "image" in example and example["image"] is not None
#         )

#     #train_dataset = dataset["train"].select(range(10)).filter(has_image)
#     train_dataset = dataset["train"].filter(has_image)
#     train_dataset = train_dataset.map(process_example, num_proc=32)


#     base_dataset_train = get_dataset(
#         train_dataset, tokenizer, processor, max_size=5000 if configs.debug else 100000000
#     )

#     total_train_steps = 0

#     if not configs.debug and rank == 0:
#         wandb_run = wandb.init(project=configs.project, name=configs.name)
#         wandb_run.config.update(configs, allow_val_change=True)
#         text_table = wandb.Table(columns=["step", "text"])

#     else:
#         wandb_run = None


#     best_acc = 0

#     collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

#     # pdb.set_trace()
#     for epoch in range(configs.resume, configs.num_epochs):

#         scheduled_stage = epoch // configs.epochs_per_stage

#         np.random.seed(epoch) 

#         dataset_train = get_cot_latent_dataset(
#             scheduled_stage,
#             base_dataset_train,
#             configs,
#             start_id,
#             latent_id,
#             end_id,
#             no_special_marker=True,
#             shuffle=True,
#         )

#         train_dataloader = torch.utils.data.DataLoader(
#             dataset_train,
#             num_workers=1,
#             shuffle=False,
#             pin_memory=True,
#             batch_size=configs.batch_size_training,
#             collate_fn=collator,
#             sampler=DistributedSampler(dataset_train, shuffle=True),
#         )

#         model_engine.train()
#         total_length = len(train_dataloader) // configs.gradient_accumulation_steps
#         pbar = tqdm(
#             colour="blue",
#             desc=f"Training Epoch: {epoch+1}",
#             total=total_length,
#             dynamic_ncols=True,
#         )
#         for step, batch in enumerate(train_dataloader):
#             print("start")
#             if step == 0 and wandb_run and rank == 0:
#                 print("logging training data")
#                 cur_bs = len(batch["input_ids"])
#                 text_str = ""
#                 for data_idx in range(cur_bs):
#                     for token_idx in range(len(batch["input_ids"][data_idx])):
#                         text_str += (
#                             str(batch["input_ids"][data_idx][token_idx].item())
#                             + " "
#                             + str(batch["labels"][data_idx][token_idx].item())
#                             + " "
#                             + tokenizer.decode(
#                                 batch["input_ids"][data_idx][token_idx]
#                             )
#                             + "\n"
#                         )
#                     text_str += "====" * 10 + "\n"

#                 text_table.add_data(total_train_steps, text_str)

#             total_train_steps += 1
#             batch = {
#                 key: batch[key].to(rank) for key in batch.keys() if key != "idx"
#             }

#             outputs = model_engine(**batch)
#             loss = outputs.loss
#             print(f"loss: {loss}")
#             model_engine.backward(loss)
#             model_engine.step()
            
#             if wandb_run and rank == 0:
#                 log_dict = {
#                     "train/epoch": epoch + 1,
#                     "train/step": epoch * len(train_dataloader) + step,
#                     "train/loss": loss.detach().float()
#                     # * configs.gradient_accumulation_steps,
#                 }
#                 wandb_run.log(log_dict)
#             # print("line432")
#             pbar.set_description(
#                 f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
#                 f"completed (loss: {round(float(loss.detach().float()), 4)}"
#             )
#             print("finish")
#         pbar.close()
#         dist.barrier()

#         if (
#             not configs.debug
#             and (epoch + 1) % 4 == 0
#         ):
            
#             epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_checkpoint")

#             model_engine.save_checkpoint(
#                 save_dir=epoch_save_dir,
#                 tag=f"epoch_{epoch+1}_zero3_bf32",
#                 client_state={"best_acc": best_acc, "current_epoch": epoch+1}
#             )

#             if rank == 0:
#                 fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_save_dir, tag=f"epoch_{epoch+1}_zero3_bf32")
#                 fp32_output = os.path.join(save_dir, f"epoch_{epoch+1}_full_model_fp32.pth")

#                 torch.save(fp32_state_dict, fp32_output)
                
#                 print(f"Epoch {epoch+1} FP32 save to {fp32_output}")

#                 if os.path.exists(epoch_save_dir):
#                     shutil.rmtree(epoch_save_dir)

#             dist.barrier()
#             gc.collect()
#             torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()


import torch
from torch_npu.contrib import transfer_to_npu
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import timedelta
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.optim import AdamW
import shutil
import numpy as np
from torch.utils.data import Subset
from collections import OrderedDict
import re
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import logging
logging.basicConfig(
    filename='qwenvl4.log',  
    level=logging.DEBUG,          
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S' 
)

from qwen_ivtlr import IVTLR

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import pdb
from peft import LoraConfig, get_peft_model
from dataset import (
    get_dataset,
    get_cot_latent_dataset,
    MyCollator,
)
from expert_runtime import build_expert_runtime_from_config
import re
from dataclasses import replace
from teacher_online import infer_teacher_dim_from_source

# LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["embed_tokens", "lm_head"], # 🌟 新增这一行！非常关键！
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)






def process_m3cot_example(example,processor):
    #数据划分，除了完整cot，分为3个部分，均匀划分
    rationale = example["rationale"].replace("\n", " ").strip()
    example["steps"] = rationale.split(". ")
    if example["steps"][-1] == "":
        example["steps"].pop()

    if len(example["steps"]) > 3:
        total_steps = len(example["steps"])
        step_size = total_steps // 3
        remainder = total_steps % 3

        new_steps = []
        start = 0

        for i in range(3):
            end = start + step_size + (1 if i < remainder else 0)
            new_steps.append(". ".join(example["steps"][start:end]))
            start = end

        example["steps"] = new_steps


    question = example["question"]
    choices = example["choices"]
    
    #两个大括号 {{ ：表示一个普通的大括号字符 { 在 f-string 中，单括号 { 有特殊含义（用于插入变量）。
    #如果你想在最终生成的字符串里显示一个真正的 {，你就必须连续写两次进行“转义”。
    choices_str = "[Options]:\n"+"\n".join([
        f"({chr(65 + i)}).{{{choice.strip()}}}"
        for i, choice in enumerate(choices)
    ])
    question = question
    question_with_braces = f"{{{question.strip()}}}"
    prefix_str = "Answer:"
    
    example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"
    
    del example["rationale"]
    del example["choices"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": example["question"]}
        ]
    }]

    example["question"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[example["question"]],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.tolist() for k, v in inputs.items()}
    example["input_ids"] = torch.tensor(inputs["input_ids"][0])
    example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    example["pixel_values"] = torch.tensor(inputs["pixel_values"])

    return example




def process_sqa_example(example,processor):
    example["answer"] = str(example["answer"])
    # rationale：merge lecture and solution
    lecture = example.get("lecture", "") or ""
    solution = example.get("solution", "") or ""
    
    # merge lecture and solution
    if lecture and solution:
        rationale = (lecture.strip() + " " + solution.strip()).strip()
    elif lecture:
        rationale = lecture.strip()
    elif solution:
        rationale = solution.strip()
    else:
        # both is null
        rationale = example["answer"]
        print(f"Warning: Both lecture and solution are empty for question: {example['question']}")
        rationale = str(rationale)

    #移除末尾空值
    rationale = rationale.replace("\n", " ").strip()
    example["steps"] = rationale.split(". ")
    if example["steps"][-1] == "":
        example["steps"].pop()

    # multi-steps，各个steps合并（均匀）
    if len(example["steps"]) > 3:
        total_steps = len(example["steps"])
        
        step_size = total_steps // 3
        remainder = total_steps % 3

        
        new_steps = []
        start = 0
       
        for i in range(3):
            end = start + step_size + (1 if i < remainder else 0)
            new_steps.append(". ".join(example["steps"][start:end]))
            start = end

        example["steps"] = new_steps
    
    question = example["question"]
    choices = example["choices"]
    choices_str = "[Options]:\n"+"\n".join([
        f"({chr(65 + i)}).{{{choice.strip()}}}"  
        for i, choice in enumerate(choices)
    ])

    question = question
    question_with_braces = f"{{{question.strip()}}}"
    prefix_str = "Answer:"
    
    example["question"] = f"[Question]:{question_with_braces}\n{choices_str}\n{prefix_str}\n"

    
    # 4. 构建 Qwen2-VL 的输入消息
    # 注意：ScienceQA 的 image 可能是 PIL 对象，process_vision_info 和 processor 可以直接处理
    messages = [{
        "role": "user",
        "content": [
            # 保持与 M3CoT 一致的 resize 策略，或者根据显存情况去掉 resize
            {"type": "image", "image": example["image"], "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": example["question"]}
        ]
    }]
    
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.tolist() for k, v in inputs.items()}
    example["input_ids"] = torch.tensor(inputs["input_ids"][0])
    example["image_grid_thw"] = torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    example["pixel_values"] = torch.tensor(inputs["pixel_values"])
    
    # 删除不需要的列
    if "lecture" in example:
        del example["lecture"]
    if "solution" in example:
        del example["solution"]
    if "choices" in example:
        del example["choices"]

        # 确保 answer 是字符串，供 dataset.py 使用
    example["answer"] = str(example["answer"]) # SQA answer 是 int (索引)，需要转 string 吗？
    # 注意：dataset.py 中 tokenize_sample 会用到 "Therefore, the answer is " + sample["answer"]
    # 对于 SQA，answer 通常是 0, 1, 2... 对应的 A, B, C?
    # Chameleon 代码中做了转换： sample["answer"] = str(sample["answer"]) 
    # 并且 extract answer 时是找 0, 1 或 A, B. 
    # 这里建议将 int index 转为 A/B/C 可能会更好，或者保持 int string。
    # 考虑到 dataset.py 代码： tokenizer.encode("Therefore, the answer is " + sample["answer"])
    # 如果 sample["answer"] 是 '0'，模型学到的就是 "...is 0"。如果是 'A'，就是 "...is A"。
    # 建议转为选项字母以保持与 Options 一致：
    if isinstance(example["answer"], int):
        example["answer"] = chr(65 + example["answer"]) # 0 -> A, 1 -> B
        
    return example





def main():
    # deepspeed初始化
    
    print("Initializing DeepSpeed Training!")
    parser = argparse.ArgumentParser(description="ivtlr")
    parser.add_argument("config")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    print("line 57")
    # load the configuration file
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    # config读取
    configs = Config(config_dict)

    # ================= [新增代码] 开始 =================
    # 动态构建实验名称，格式为: {数据集}_qwen2vl_{模型大小}_IVTLR
    # 这样每次修改 yaml 中的 dataset_name 或 model_type，输出目录会自动更新
    dataset_name = getattr(configs, "dataset_name", "m3cot")
    model_type = getattr(configs, "model_type", "7B")
    suffix = getattr(configs, "suffix", "")
    
    new_name = f"{dataset_name}_qwen2vl_{model_type}_IVTLR{suffix}" 
    
    # 如果想保留 yaml 里自定义的 name 前缀，也可以做拼接，这里建议直接覆盖以保证规范
    if configs.name != new_name:
        print(f"[Config Auto-Update] Changing experiment name from '{configs.name}' to '{new_name}'")
        configs.name = new_name
    # ================= [新增代码] 结束 =================

    
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    cur_ckpts = os.listdir(save_dir)


    # check if the job is preempted and resumed.
    if len(cur_ckpts) > 0 and rank == 0:
        raise ValueError(
            f"Save directory {save_dir} is not empty! "
        )

    if configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )
        
        
    # 模型、分词器加载，新增token    
    print("start loading model")
    # [修改] 解析模型路径
    target_model_type = getattr(configs, "model_type", "7B") # 默认为 7B
    if target_model_type == "2B":
        model_path = getattr(configs, "model_path_2b", "/home/ma-user/work/lbx/models/Qwen2-VL-2B-Instruct")
    else:
        model_path = getattr(configs, "model_path_7b", "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct")
    
    print(f"Selected Model: {target_model_type}, Path: {model_path}")
    
    # Todo:modify model and Tokenizer
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.lr)
    #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # 【新增】：在添加特殊 token 之前，记录原始模型词表大小
    original_vocab_size = len(tokenizer)
    
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    # Expert runtime (adds expert tokens as normal tokens so decode() keeps them)
    expert_runtime = build_expert_runtime_from_config(
        config_dict,
        tokenizer,
        add_tokens=True,
        for_inference=False,
    )
    if rank == 0 and expert_runtime is not None:
        print(f"[Expert] enabled={expert_runtime.enabled}, experts={expert_runtime.experts}, slots={expert_runtime.slots}")

    # Infer teacher dims automatically if not provided in YAML.
    if expert_runtime is not None and expert_runtime.align_enabled and expert_runtime.align_weight > 0:
        dims = dict(expert_runtime.teacher_dims)
        for e in expert_runtime.experts:
            if int(dims.get(e, 0)) > 0:
                continue

            if expert_runtime.teacher_mode in ("online", "hybrid"):
                src = expert_runtime.teacher_sources.get(e)
                if not src:
                    raise ValueError(f"Missing expert.teacher.online.models.{e} (required to infer dims in {expert_runtime.teacher_mode} mode)")
                dims[e] = int(infer_teacher_dim_from_source(e, src))
            elif expert_runtime.teacher_mode == "offline":
                # Infer from any cache entry (if present); otherwise require explicit dims.
                cache_dir = expert_runtime.teacher_cache_dir
                if not cache_dir:
                    raise ValueError("Missing expert.teacher.offline.cache_dir (required for offline mode)")
                exp_dir = os.path.join(cache_dir, e)
                dim_found = 0
                if os.path.isdir(exp_dir):
                    for fn in os.listdir(exp_dir):
                        if fn.endswith(".pt"):
                            vec = torch.load(os.path.join(exp_dir, fn), map_location="cpu")
                            if torch.is_tensor(vec):
                                dim_found = int(vec.view(-1).numel())
                                break
                if dim_found <= 0:
                    raise ValueError(
                        f"Cannot infer teacher dim for offline expert={e} from cache_dir={cache_dir}. "
                        f"Please add at least one cached vector or set expert.teacher_dims.{e}."
                    )
                dims[e] = dim_found
            else:
                raise ValueError(f"Unknown expert.teacher.mode: {expert_runtime.teacher_mode}")

        expert_runtime = replace(expert_runtime, teacher_dims=dims)
        if rank == 0:
            print(f"[Expert] inferred teacher_dims={expert_runtime.teacher_dims}")

    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", tokenizer=tokenizer)
    processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    print("latent_id: ", latent_id)
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

   

    loaded = False

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    # it helps stablize the training
    # 这里似乎没有使用已有的embedding初始化？
    init_token_ids = [latent_id, start_id, end_id]
    if expert_runtime is not None:
        init_token_ids.extend(expert_runtime.token_id_list())
    for token_id in init_token_ids:
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding
        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    
    
    # ================= 【新增核心代码】 =================
    # 定义梯度 Hook 函数：把前 original_vocab_size 个词的梯度强制置 0
    def freeze_original_embeddings_hook(grad):
        # 必须 clone，避免 inplace 操作报错
        grad_clone = grad.clone()
        # 将原始词表部分的梯度清零
        grad_clone[:original_vocab_size, :] = 0
        return grad_clone

    # 给 Input Embedding 的权重挂上 Hook
    if embeddings.weight.requires_grad:
        embeddings.weight.register_hook(freeze_original_embeddings_hook)
        print(f"已锁定前 {original_vocab_size} 个 Input Embeddings 不更新")

    # 给 Output LM Head 的权重挂上 Hook
    if model.lm_head.weight.requires_grad:
        model.lm_head.weight.register_hook(freeze_original_embeddings_hook)
        print(f"已锁定前 {original_vocab_size} 个 LM Head 不更新")
    # ===================================================
    
     #  LoRA配置和token嵌入初始化
    model = get_peft_model(model, lora_config)
    
    
    model.print_trainable_parameters()

    model = IVTLR(
        model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        model_path=model_path,
        expert_runtime=expert_runtime,
    )

    # deepspeed包装模型
    print(f"Running Deepspeed on rank = {rank}, world size = {world_size}")
    model = model.to(rank)
    
    if configs.bf16:
        model.to(torch.bfloat16)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        # optimizer = optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters())
    )

    del model


    # 数据集加载
    # --- Dataset Loading Logic Modified ---
    print("start dataset")
    dataset_name = getattr(configs, "dataset_name", "m3cot") # 默认为 m3cot 以兼容旧配置
    print(f"Loading dataset: {dataset_name}")

    # 处理 train 数据集
    print("start dataset")
    

    if dataset_name == "scienceqa":
        dataset = load_dataset("derek-thomas/ScienceQA")
        
        def has_image(example):
            return "image" in example and example["image"] is not None
        
        # 选取部分数据用于调试或全量
        if configs.debug:
            train_dataset = dataset["train"].select(range(40)).filter(has_image)
        else:
            train_dataset = dataset["train"].filter(has_image)
            
        # 使用 partial 传递 processor
        process_func = functools.partial(process_sqa_example, processor=processor)
        train_dataset = train_dataset.map(process_func, num_proc=32,)
    
    elif dataset_name == "m3cot":
        dataset = load_dataset("LightChen2333/M3CoT")
        
        def has_image(example):
            return "image" in example and example["image"] is not None

        if configs.debug:
            train_dataset = dataset["train"].select(range(40)).filter(has_image)
        else:
            train_dataset = dataset["train"].filter(has_image)
            
        process_func = functools.partial(process_m3cot_example, processor=processor)
        train_dataset = train_dataset.map(process_func, num_proc=32)
        
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

        

    keep_image = False
    if expert_runtime is not None and getattr(expert_runtime, "enabled", False):
        if expert_runtime.teacher_mode in ("online", "hybrid") and "sam" in expert_runtime.experts:
            keep_image = True

    base_dataset_train = get_dataset(
        train_dataset,
        tokenizer,
        processor,
        max_size=5000 if configs.debug else 100000000,
        keep_image=keep_image,
    )

    # Optional: quick overfit/format validation mode driven by YAML.
    fast_cfg = None
    if isinstance(config_dict.get("expert"), dict):
        fast_cfg = config_dict["expert"].get("format_overfit_test")
        if not isinstance(fast_cfg, dict):
            fast_cfg = None
        elif not bool(fast_cfg.get("enabled", False)):
            fast_cfg = None

    num_epochs_to_run = configs.num_epochs
    stage_override = None
    if fast_cfg is not None:
        ds_size = int(fast_cfg.get("dataset_size", 16))
        if ds_size > 0:
            base_dataset_train = base_dataset_train.select(range(min(ds_size, len(base_dataset_train))))
            if rank == 0:
                print(f"[FastCheck] Using dataset_size={len(base_dataset_train)}")
        num_epochs_to_run = int(fast_cfg.get("num_epochs", num_epochs_to_run))
        stage_override = str(fast_cfg.get("stage", "")).strip().lower() or None
        if rank == 0:
            print(f"[FastCheck] num_epochs={num_epochs_to_run}, stage_override={stage_override}")

    def _run_format_check(model_for_gen, check_samples: int = 8):
        if expert_runtime is None or not getattr(expert_runtime, "enabled", False):
            print("[FastCheck] expert_runtime is disabled; skip format check.")
            return

        model_for_gen.eval()
        n = min(int(check_samples), len(base_dataset_train))
        print(f"[FastCheck] Running format check on {n} samples from training data.")

        for i in range(n):
            ex = base_dataset_train[i]
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

            # Build inference prompt from training sample: question + latent tokens (no steps/answer).
            if stage_override == "skip_all_steps":
                if configs.pad_latent_to_max:
                    n_latent = int(configs.max_latent_stage)
                else:
                    n_latent = min(len(ex["steps_tokenized"]), int(configs.max_latent_stage))
            else:
                # Fallback: mirror stage=0 prompt by default.
                n_latent = 0

            prompt_ids = ex["question_tokenized"] + [latent_id] * int(n_latent)
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device_id).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            pv = torch.as_tensor(ex["pixel_values"]).to(device_id)
            if pv.ndim == 3:
                pv = pv.unsqueeze(0)
            grid = torch.as_tensor(ex["image_grid_thw"]).to(device_id)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)

            with torch.no_grad():
                out_ids = model_for_gen.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pv,
                    image_grid_thw=grid,
                    max_new_tokens=256,
                )

            gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            ans_pos = gen.lower().find("therefore")
            ok = True
            for e in expert_runtime.experts:
                tok = expert_runtime.token_strs[e]
                pos = gen.find(tok)
                if pos < 0:
                    ok = False
                if ans_pos >= 0 and pos >= 0 and pos > ans_pos:
                    ok = False
            print(f"[FastCheck] sample={i} ok={ok}")
            # if not ok:
            print(gen[-600:])


    total_train_steps = 0

    if not configs.debug and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None


    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    
    # Preview stage formats before training starts.
    if expert_runtime is not None:  # 🔥 去掉 rank == 0 的限制，让所有卡都参与生成数据
        try:
            stages = sorted({e // configs.epochs_per_stage for e in range(configs.num_epochs)})
            preview_stages = stages + [configs.max_latent_stage + 1]
            preview_stages = sorted(set(preview_stages))
            if rank == 0:
                print(f"[Stage Preview] stages={preview_stages}")
                
            for st in preview_stages:
                # 所有 GPU 都会调用这个函数，顺利完成底层的 broadcast 同步
                ds_preview = get_cot_latent_dataset(
                    st,
                    base_dataset_train.select(range(1)),
                    configs,
                    start_id,
                    latent_id,
                    end_id,
                    expert_runtime=expert_runtime,
                    no_special_marker=True,
                    shuffle=False,
                )
                
                # 🔥 但是只让 0 号卡负责打印预览结果
                if rank == 0:
                    ex = ds_preview[0]
                    ids = ex["input_ids"]
                    lbl = ex["labels"]
                    decoded = tokenizer.decode(ids, skip_special_tokens=False)
                    print("=" * 40)
                    print(f"[Stage {st}] input_len={len(ids)} labels_len={len(lbl)}")
                    ce_start = next((i for i, v in enumerate(lbl) if v != -100), None)
                    print(f"[Stage {st}] ce_start={ce_start}")
                    for e in expert_runtime.experts:
                        tid = expert_runtime.token_ids[e]
                        pos = [i for i, x in enumerate(ids) if x == tid]
                        print(f"[Stage {st}] expert={e} token_pos={pos}")
                    print(decoded[:])
        except Exception as e:
            if rank == 0:
                print(f"[Stage Preview] failed: {e}")
    # # Preview stage formats before training starts.
    # if rank == 0 and expert_runtime is not None:
    #     try:
    #         stages = sorted({e // configs.epochs_per_stage for e in range(configs.num_epochs)})
    #         preview_stages = stages + [configs.max_latent_stage + 1]
    #         preview_stages = sorted(set(preview_stages))
    #         print(f"[Stage Preview] stages={preview_stages}")
    #         for st in preview_stages:
    #             ds_preview = get_cot_latent_dataset(
    #                 st,
    #                 base_dataset_train.select(range(1)),
    #                 configs,
    #                 start_id,
    #                 latent_id,
    #                 end_id,
    #                 expert_runtime=expert_runtime,
    #                 no_special_marker=True,
    #                 shuffle=False,
    #             )
    #             ex = ds_preview[0]
    #             ids = ex["input_ids"]
    #             lbl = ex["labels"]
    #             decoded = tokenizer.decode(ids, skip_special_tokens=False)
    #             print("=" * 40)
    #             print(f"[Stage {st}] input_len={len(ids)} labels_len={len(lbl)}")
    #             # CE positions: where label != -100
    #             ce_start = next((i for i, v in enumerate(lbl) if v != -100), None)
    #             print(f"[Stage {st}] ce_start={ce_start}")
    #             for e in expert_runtime.experts:
    #                 tid = expert_runtime.token_ids[e]
    #                 pos = [i for i, x in enumerate(ids) if x == tid]
    #                 print(f"[Stage {st}] expert={e} token_pos={pos}")
    #             print(decoded[-400:])
    #     except Exception as e:
    #         print(f"[Stage Preview] failed: {e}")

    # 训练循环
    # pdb.set_trace()
    for epoch in range(configs.resume, num_epochs_to_run):

        # 正在处于训练的哪一个阶段
        if stage_override == "skip_all_steps":
            scheduled_stage = int(configs.max_latent_stage) + 1
        else:
            scheduled_stage = epoch // configs.epochs_per_stage

        np.random.seed(epoch) 
        # import torch
        # 得到该阶段的数据集
        dataset_train = get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_train,
            configs,
            start_id,
            latent_id,
            end_id,
            expert_runtime=expert_runtime,
            no_special_marker=True,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_train, shuffle=True),
        )

        
        # ================= [新增：每个 Epoch 开始前推理验证 1 个样本] =================
        if expert_runtime is not None and getattr(expert_runtime, "enabled", False):
            if rank == 0:
                print(f"\n========== Epoch {epoch+1} 开始：格式遵循实时验证 ==========")
            try:
                # 所有进程共同参与 generate (DeepSpeed ZeRO-3 必须要求)，只测 1 个样本
                _run_format_check(model_engine.module, check_samples=1)
            except Exception as e:
                if rank == 0:
                    print(f"[Epoch {epoch+1} FastCheck] format check failed: {e}")
            if rank == 0:
                print("============================================================\n")
        # ==============================================================================
        
        model_engine.train()
        # total_length是实际参数更新的次数
        total_length = len(train_dataloader) // configs.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            print("start")
#             只在第一个batch记录详细数据
#             用于调试和可视化训练数据
#             显示token ID、label和解码文本的对应关系
            if step == 0 and wandb_run and rank == 0:
                print("logging training data")
                cur_bs = len(batch["input_ids"])
                text_str = ""
                for data_idx in range(cur_bs):
                    for token_idx in range(len(batch["input_ids"][data_idx])):
                        text_str += (
                            str(batch["input_ids"][data_idx][token_idx].item())
                            + " "
                            + str(batch["labels"][data_idx][token_idx].item())
                            + " "
                            + tokenizer.decode(
                                batch["input_ids"][data_idx][token_idx]
                            )
                            + "\n"
                        )
                    text_str += "====" * 10 + "\n"
                # log
                text_table.add_data(total_train_steps, text_str)
                wandb_run.log({"training_samples": text_table})
                print(f'第{total_train_steps}步:\n{text_str}')

                
            total_train_steps += 1
            batch = {key: (batch[key].to(rank) if torch.is_tensor(batch[key]) else batch[key]) for key in batch.keys()}

            outputs = model_engine(**batch)
            # 【新增】安全获取 expert_losses 字典
            expert_losses = getattr(outputs, "expert_losses", {})
            loss = outputs.loss
            if rank == 0:
                print(f"loss: {loss.item():.4f}")
                for e_name, e_loss in expert_losses.items():
                    print(f"  expert_{e_name}_loss: {e_loss.item():.4f}")
                    
            
           # === 测试点：记录更新前的权重 ===
            old_weight = model_engine.module.base_causallm.base_model.model.model.language_model.layers[27].mlp.up_proj.lora_B.default.weight.detach().clone()
            # old_weight2 =  model_engine.module.head_gate[0].weight.detach().clone()
            # old_weight3 =  model_engine.module.layer_gate[0].weight.detach().clone()
            model_engine.backward(loss)
            model_engine.step()
            
            # === 测试点：检查权重是否更新 ===
            new_weight = model_engine.module.base_causallm.base_model.model.model.language_model.layers[27].mlp.up_proj.lora_B.default.weight.detach().clone()
            # new_weight2 =  model_engine.module.head_gate[0].weight.detach().clone()
            # new_weight3 =  model_engine.module.layer_gate[0].weight.detach().clone()
            diff = (new_weight - old_weight).abs().sum().item()
            # diff2 = (new_weight2 - old_weight2).abs().sum().item()
            # diff3 = (new_weight3 - old_weight3).abs().sum().item()
            if rank == 0:
                print(f"layers.27.mlp.up_proj.lora_B.default.weight 权重变化量: {diff}")
                # print(f"head_gate_weight 权重变化量: {diff2}")
                # print(f"layer_gate.weight 权重变化量: {diff3}")
            
            if wandb_run and rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": epoch * len(train_dataloader) + step,
                    "train/loss": loss.detach().float()
                    # * configs.gradient_accumulation_steps,
                }
                # 【新增】将每个专家的 loss 记录到 Wandb
                for e_name, e_loss in expert_losses.items():
                    log_dict[f"train/expert_{e_name}_loss"] = e_loss.float()
                    
                wandb_run.log(log_dict)

            # print("line432")
            pbar.set_description(
                f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                f"completed (loss: {round(float(loss.detach().float()), 4)}"
            )
            print("finish")
        pbar.close()
        dist.barrier()

        # 每个stage结束后，保存一次模型权重
        #           # not configs.debug and 
            
        if (
             (epoch + 1) % 4 == 0
        ):
            
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_checkpoint")

            model_engine.save_checkpoint(
                save_dir=epoch_save_dir,
                tag=f"epoch_{epoch+1}_zero3_bf32",
                client_state={"best_acc": best_acc, "current_epoch": epoch+1}
            )

            if rank == 0:
                fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_save_dir, tag=f"epoch_{epoch+1}_zero3_bf32")
                fp32_output = os.path.join(save_dir, f"epoch_{epoch+1}_full_model_fp32.pth")

                torch.save(fp32_state_dict, fp32_output)
                
                print(f"Epoch {epoch+1} FP32 save to {fp32_output}")

                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

    # Post-training quick format check using training data as inference content.
    if fast_cfg is not None:  # 🔥 同样去掉 rank == 0
        try:
            _run_format_check(model_engine.module, check_samples=int(fast_cfg.get("check_samples", 4)))
        except Exception as e:
            if rank == 0:
                print(f"[FastCheck] format check failed: {e}")

if __name__ == "__main__":
    main()
