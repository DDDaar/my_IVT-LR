import functools
import re
from typing import Dict, List

from datasets import Dataset, load_dataset


def _has_image(example: Dict) -> bool:
    return "image" in example and example["image"] is not None


def _split_rationale_to_steps(rationale: str, max_chunks: int = 3) -> List[str]:
    rationale = (rationale or "").replace("\n", " ").strip()
    steps = rationale.split(". ")
    if steps and steps[-1] == "":
        steps.pop()
    if not steps:
        return [""]

    if len(steps) <= max_chunks:
        return steps

    total_steps = len(steps)
    step_size = total_steps // max_chunks
    remainder = total_steps % max_chunks
    merged = []
    start = 0
    for i in range(max_chunks):
        end = start + step_size + (1 if i < remainder else 0)
        merged.append(". ".join(steps[start:end]))
        start = end
    return merged


def _format_question(question: str, choices: List[str]) -> str:
    """
    Keep exactly the same formatting style as SFT preprocessing:
    [Question]:{...}
    [Options]:
    (A).{...}
    ...
    Answer:
    """
    choices_str = "[Options]:\n" + "\n".join(
        [f"({chr(65 + i)}).{{{str(choice).strip()}}}" for i, choice in enumerate(choices)]
    )
    question_with_braces = f"{{{str(question).strip()}}}"
    return f"[Question]:{question_with_braces}\n{choices_str}\nAnswer:\n"


def _normalize_answer_to_letter(answer, choices: List[str]) -> str:
    if isinstance(answer, int):
        idx = int(answer)
        if 0 <= idx < len(choices):
            return chr(65 + idx)

    answer_str = str(answer).strip()
    match = re.match(r"^\(?([A-Za-z])\)?$", answer_str)
    if match:
        return match.group(1).upper()

    answer_lower = answer_str.lower()
    for idx, choice in enumerate(choices):
        if str(choice).strip().lower() == answer_lower:
            return chr(65 + idx)

    return answer_str.upper()


def _latent_suffix(n_latent_tokens: int) -> str:
    if n_latent_tokens <= 0:
        return ""
    return "".join(["<|latent|>"] * n_latent_tokens)


def _build_train_row(example: Dict, processor, latent_tokens_per_sample: int, dataset_name: str) -> Dict:
    choices = example["choices"]
    question_text = _format_question(example["question"], choices)

    # Keep raw conversational structure for TRL multimodal GRPO.
    # Do not pre-apply chat template here; trainer will apply it internally.
    prompt_text = f"{question_text}{_latent_suffix(latent_tokens_per_sample)}\n"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    image = example["image"]
    if image is not None and getattr(image, "mode", "RGB") != "RGB":
        image = image.convert("RGB")

    valid_letters = ",".join([chr(65 + i) for i in range(len(choices))])
    answer_letter = _normalize_answer_to_letter(example["answer"], choices)

    return {
        "prompt": messages,
        "image": image,
        "ground_truth": answer_letter,
        "valid_letters": valid_letters,
        "dataset_name": dataset_name,
    }


def _prepare_m3cot(example: Dict) -> Dict:
    example["steps"] = _split_rationale_to_steps(example.get("rationale", ""))
    return example


def _prepare_scienceqa(example: Dict) -> Dict:
    lecture = example.get("lecture", "") or ""
    solution = example.get("solution", "") or ""
    rationale = ""
    if lecture and solution:
        rationale = f"{lecture.strip()} {solution.strip()}".strip()
    elif lecture:
        rationale = lecture.strip()
    elif solution:
        rationale = solution.strip()
    else:
        rationale = str(example["answer"])

    example["steps"] = _split_rationale_to_steps(rationale)
    return example


def build_grpo_dataset(
    dataset_name: str,
    processor,
    k_samples: int,
    seed: int,
    max_latent_stage: int,
    pad_latent_to_max: bool,
    num_proc: int = 1,
) -> Dataset:
    dataset_name = dataset_name.lower()

    if dataset_name == "m3cot":
        raw = load_dataset("LightChen2333/M3CoT")["train"]
        raw = raw.filter(_has_image, num_proc=num_proc)
    elif dataset_name == "scienceqa":
        raw = load_dataset("derek-thomas/ScienceQA")["train"]
        raw = raw.filter(_has_image, num_proc=num_proc)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    raw = raw.shuffle(seed=seed)
    if k_samples > 0:
        raw = raw.select(range(min(k_samples, len(raw))))

    # Keep all expensive map steps strictly on the selected subset only.
    if not pad_latent_to_max:
        if dataset_name == "m3cot":
            raw = raw.map(_prepare_m3cot, num_proc=num_proc, load_from_cache_file=False)
        elif dataset_name == "scienceqa":
            raw = raw.map(_prepare_scienceqa, num_proc=num_proc, load_from_cache_file=False)

    if pad_latent_to_max:
        latent_tokens_per_sample = int(max_latent_stage)
        mapper = functools.partial(
            _build_train_row,
            processor=processor,
            latent_tokens_per_sample=latent_tokens_per_sample,
            dataset_name=dataset_name,
        )
        columns = list(raw.features)
        return raw.map(
            mapper,
            remove_columns=columns,
            num_proc=num_proc,
            load_from_cache_file=False,
        )

    def dynamic_mapper(example):
        n_latent = min(len(example.get("steps", [])), int(max_latent_stage))
        return _build_train_row(
            example=example,
            processor=processor,
            latent_tokens_per_sample=n_latent,
            dataset_name=dataset_name,
        )

    columns = list(raw.features)
    return raw.map(
        dynamic_mapper,
        remove_columns=columns,
        num_proc=num_proc,
        load_from_cache_file=False,
    )
