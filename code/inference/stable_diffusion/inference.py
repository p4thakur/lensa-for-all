# Here are some references.
# https://huggingface.co/docs/diffusers/stable_diffusion
# https://huggingface.co/docs/diffusers/optimization/fp16

import base64
from io import BytesIO
from typing import Any, Dict, List, Union
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


def model_fn(model_dir: str) -> Any:
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=True,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # pipe.enable_vae_tiling()
    # pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


def predict_fn(
    input_data: Dict[str, Union[int, float, str]], pipe: Any
) -> Dict[str, List[str]]:
    prompt = input_data.pop("prompt", input_data)
    negative_prompt = input_data.pop("negative_prompt", None)
    num_inference_steps = input_data.pop("num_inference_steps", 50)
    guidance_scale = input_data.pop("guidance_scale", 7.5)
    num_images_per_prompt = input_data.pop("num_images_per_prompt", 4)
    width = input_data.pop("width", 512)
    height = input_data.pop("height", 512)

    if negative_prompt is None or len(negative_prompt) == 0:
        kwargs = {}
    else:
        kwargs = {"negative_prompt": negative_prompt}

    generated_images = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        width=width,
        height=height,
        **kwargs,
    )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"generated_images": encoded_images, "prompt": prompt}
