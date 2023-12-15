# TokenFlow: Consistent Diffusion Features for Consistent Video Editing

## [<a href="https://diffusion-tokenflow.github.io/" target="_blank">Project Page</a>]

[![arXiv](https://img.shields.io/badge/arXiv-TokenFlow-b31b1b.svg)](https://arxiv.org/abs/2307.10373) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/weizmannscience/tokenflow)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch)


[//]: # ([![Replicate]&#40;https://replicate.com/cjwbw/multidiffusion/badge&#41;]&#40;https://replicate.com/cjwbw/multidiffusion&#41;)

[//]: # ([![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/weizmannscience/text2live&#41;)




https://github.com/omerbt/TokenFlow/assets/52277000/93dccd63-7e9a-4540-a941-31962361b0bb

**TokenFlow** is a framework that enables consistent video editing, using a pre-trained text-to-image diffusion model,
without any further training or finetuning.

## 🚨 Updates 🚨
 - **DDPM inversion:** The latest version now supports ddpm inversion, providing up to 95% reduced inversion times and better reconstruction quality. </br> 
DDPM is now set as default. **Directory structure has been updated with this change.**

[//]: # (as described in <a href="https://arxiv.org/abs/2302.08113" target="_blank">&#40;link to paper&#41;</a>.)

[//]: # (. It can be used for localized and global edits that change the texture of existing objects or augment the scene with semi-transparent effects &#40;e.g. smoke, fire, snow&#41;.)

[//]: # (### Abstract)
> The generative AI revolution has been recently expanded to videos. Nevertheless, current state-of-the-art video mod-
> els are still lagging behind image models in terms of visual quality and user control over the generated content. In
> this work, we present a framework that harnesses the power of a text-to-image diffusion model for the task of
> text-driven video editing. Specifically, given a source video and a target text-prompt, our method generates a
> high-quality video that adheres to the target text, while preserving the spatial layout and dynamics of the input
> video. Our method is based on our key observation that consistency in the edited video can be obtained by enforcing
> consistency in the diffusion feature space. We achieve this by explicitly propagating diffusion features based on
> inter-frame correspondences, readily available in the model. Thus, our framework does not require any training or
> fine-tuning, and can work in conjunction with any off-the-shelf text-to-image editing method. We demonstrate
> state-of-the-art editing results on a variety of real-world videos.

For more see the [project webpage](https://diffusion-tokenflow.github.io).

## Sample results

<td><img src="assets/videos.gif"></td>

## Environment

```
conda create -n tokenflow python=3.9
conda activate tokenflow
pip install -r requirements.txt
```

## Preprocess

Preprocess you video by running using the following command:

```
python preprocess.py --data_path <data/myvideo.mp4> \
                     --inversion_prompt <'' or a string describing the video content>
```

Additional arguments:

```                     
                     --H <video height, defaults to 512>
                     --W <video width, defaults to 512>
                     --save_dir <defaults to 'latents'>
                     --sd_version <Stable-Diffusion version, defaults to 2.1>
                     --reconstruct <will reconstruct the original video if flag is set>
                     --steps <number of inversion steps. Should be between 50 ad 100 for DDPM and 500 and 1000 for DDIM inversion, respectively. defaults to 50>
                     --batch_size <batch size used, defaults to 40>
                     --save_steps <number of sampling steps that will be used later for editing. Ignored for DDPM inversion.>
                     --inversion <number of frames, defaults to 40>
                     --skip_steps <Initial diffusion steps to skip at inference>
                     
                     
```

more information on the arguments can be found here.

### Note:

Contrary to DDIM, edit-friendly DDPM inversion guarantees perfect reconstruction for any number of timesteps, thus eliminating reconstruction errors. 
Consequently, the number of inversion steps should be chosen equal to the number of inference steps

## Editing

- TokenFlow is designed for video for structure-preserving edits.
- Our method is built on top of an image editing technique (e.g., Plug-and-Play, ControlNet, etc.) - therefore, it is
  important to ensure that the edit works with the chosen base technique.
- The LDM decoder may introduce some jitter, depending on the original video.

To edit your video, first create a yaml config as in ``configs/config_pnp.yaml``.
Then run

```
python run_tokenflow_pnp.py
```

Similarly, if you want to use ControlNet or SDEdit, create a yaml config as in ``config/config_controlnet.yaml``
or ```configs/config_SDEdit.yaml``` and run ```python run_tokenflow_controlnet.py```
or ``python run_tokenflow_SDEdit.py`` respectively.

## Citation

```
@article{tokenflow2023,
        title = {TokenFlow: Consistent Diffusion Features for Consistent Video Editing},
        author = {Geyer, Michal and Bar-Tal, Omer and Bagon, Shai and Dekel, Tali},
        journal={arXiv preprint arxiv:2307.10373},
        year={2023}
        }
```

