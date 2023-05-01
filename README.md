# Lensa-for-All
- - -
![avatars-01](assets/avatars-01.png)
This is **Lensa for All**, an application that creates personalized images by fine-tuning the image generation model with just a few photos. If you have 4-20 photos and an AWS account ID, you can create as many [*Lensa*](https://prisma-ai.com/lensa)-style AI avatars as you want for a fraction of the price.
* The application enables the generation of images of you, your family and friends, your pets, etc. by fine-tuning the [*Stable Diffusion*](https://stability.ai/blog/stable-diffusion-public-release) using the *Dreambooth* proposed by Google in their paper ['DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (2022)'](https://dreambooth.github.io/).
* This used [*Amazon SageMaker*](https://aws.amazon.com/sagemaker/) Services and 🤗 [*Diffusers*](https://huggingface.co/docs/diffusers/index) and [*Accelerate*](https://huggingface.co/docs/accelerate/index) to optimize infrastructure requirements. The model training takes about an hour, which is the equivalent of $1.212 as of April 2023 for `ml.g5.2xlarge` in the Virginia region of the US. (It's less if you use `ml.g4dn.2xlarge`.)
* This also supports auto-prompting using [Gustavosta's *MagicPrompt*](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion) based on *GPT2*. 

## Installation
- - -
### Requirements

## Usage
- - -
1. ...

![avatars-02](assets/avatars-02.png)
## Tips for Successful Fine-tuning
- - -
* Start with a good number of high-quality training images (4 to 20). If you are training on human faces, you may need more images. I recommend [12 faces, 5 upper body shots, and 3 full body shots](https://techpp.com/2022/10/10/how-to-train-stable-diffusion-ai-dreambooth/) taken in [a variety of backgrounds, lighting, gazes, and facial expressions](https://github.com/JoePenna/Dreambooth-Stable-Diffusion).
* *Dreambooth* tends to overfit quickly, so I recommend setting the learning rate low (1e-06 to 2e-06) and gradually increasing the number of training steps until you are satisfied with the results. As a guide, if the generated images are noisy or degraded and look almost identical to the photos in your training set, they're overfitted. (If you've set up *WandB*, look at the validation images that are generated and make a judgment call. For example, if the Eiffel Tower that you set as the background in the prompt starts to disappear, that's evidence of overfitting). To address this, reduce the number of training steps. 
* When learning non-human objects, such as dogs and cats, you typically need 200 to 400 steps of training. However, if you're learning a human face, you may need more training, perhaps 800 to 1200 steps.
* If you're not training human faces, you don't need to use *Prior Preservation Loss*, and it won't have a significant impact on performance. However, you should use it for human faces training.
Also, fine-tuning the text encoder along with the image generator has been shown to produce the best image quality. However, using priors and training the text encoder requires more GPU memory (at least `ml.g5.2xlarge`). If you use techniques like [*8-bit Adam*](https://arxiv.org/abs/2110.02861), half precision training, or [gradient accumulation](https://arxiv.org/abs/1710.02368), you can get by with `ml.g4dn.2xlarge`.

## Roadmap
- - -
- [ ] Provides quality prompt templates from [*PromptHero*](https://prompthero.com/) and more.
- [ ] Enable early stopping of model training based on [*FID*](https://arxiv.org/pdf/1706.08500.pdf) scores.
- [ ] Instead of manually generating images using *Jupyter Notebook*, create a *Python* script that automatically generates the desired number of images.
- [ ] Redesign each workflow to be object-oriented, and organize and register them as a *Python* package.

## References
- - -
* [*Stable Diffusion* on *Amazon SageMaker*](https://www.philschmid.de/sagemaker-stable-diffusion)
* [Fine-Tune Text-to-Image *Stable Diffusion* Models with *Amazon SageMaker JumpStart*](https://aws.amazon.com/blogs/machine-learning/fine-tune-text-to-image-stable-diffusion-models-with-amazon-sagemaker-jumpstart/)
* [Training *Stable Diffusion* with *Dreambooth* Using 🧨 *Diffusers*](https://huggingface.co/blog/dreambooth)
* [How to Fine-tune *Stable Diffusion* Using *Dreambooth*](https://towardsdatascience.com/how-to-fine-tune-stable-diffusion-using-dreambooth-dfa6694524ae)
