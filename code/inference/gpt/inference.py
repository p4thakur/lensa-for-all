import ast
from typing import Any, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM


def model_fn(model_dir: str) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def predict_fn(
    input_data: Dict[str, Union[int, float, str]],
    model_dict: Dict[str, Any],
) -> Dict[str, Union[int, float, str]]:
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    use_magic_prompt = input_data.pop("use_magic_prompt", "False")
    use_magic_prompt = ast.literal_eval(use_magic_prompt)

    if use_magic_prompt:
        prompt = input_data.pop("prompt")
        prompt = prompt if prompt[-1] == "," else prompt + ","
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, do_sample=True)
        prompt = tokenizer.decode(output[0], skip_special_tokens=True)
        input_data["prompt"] = prompt

        return input_data
    return input_data
