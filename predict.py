# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import glob
import shutil
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
import numpy as np
import cv2
from PIL import Image
from torchvision.io import read_video, write_video
import torchvision.transforms as T
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
)
from cog import BasePredictor, Input, Path

from tokenflow_utils import *
from util import save_video, seed_everything, add_dict_to_yaml_file


VAE_BATCH_SIZE = 10


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda:0"
        self.sd_version = "2.1"
        config = argparse.Namespace(
            sd_version=self.sd_version,
            cache_dir="model_cache",
            device=self.device,
        )
        model_key = "stabilityai/stable-diffusion-2-1-base"
        self.toy_scheduler = DDIMScheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            cache_dir=config.cache_dir,
        )
        self.model = Preprocess(self.device, config)
        self.editor = TokenFlow(vars(config))

    def predict(
        self,
        video: Path = Input(description="Input video."),
        inversion_prompt: str = Input(
            description="Describe your input video or leave it empty.",
            default="",
        ),
        diffusion_prompt: str = Input(
            description="Describe your output video.",
        ),
        negative_diffusion_prompt: str = Input(
            description="Specify things to not see in the output",
            default="ugly, blurry, low res, unrealistic, unaesthetic",
        ),
        n_frames: int = Input(
            description="Number of frames in the video to process.", default=40
        ),
        width: int = Input(
            description="Width of the output video. For non-square videos, we recommend using 672 x 384 or 384 x 672, aspect ratio 1.75.",
            default=512,
        ),
        height: int = Input(
            description="Height of the output video. For non-square videos, we recommend using 672 x 384 or 384 x 672, aspect ratio 1.75.",
            default=512,
        ),
        fps: int = Input(
            description="Frames per second in the output video", default=10
        ),
        num_inversion_steps: int = Input(
            description="Number of inversion step.", default=50
        ),
        num_diffusion_steps: int = Input(
            description="Number of sampling step.", default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        seed_everything(seed)

        opt = argparse.Namespace(
            data_path=str(video),
            H=height,
            W=width,
            sd_version=self.sd_version,
            steps=num_inversion_steps,
            batch_size=40,
            save_steps=50,
            n_frames=n_frames,
            inversion_prompt=inversion_prompt,
        )

        experiment_dir = "experiment_dir"
        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir)

        opt.data_path = os.path.join(experiment_dir, "input_frames")
        save_intput_video_frames(str(video), opt.data_path, img_size=(opt.W, opt.H))

        self.toy_scheduler.set_timesteps(opt.save_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(
            self.toy_scheduler,
            num_inference_steps=opt.save_steps,
            strength=1.0,
        )

        os.makedirs(os.path.join(experiment_dir, "latents"))
        paths, frames, max_frames = get_data(opt.data_path, opt.n_frames, self.device)
        # encode to latents
        latents = (
            self.model.encode_imgs(frames, deterministic=True)
            .to(torch.float16)
            .to(self.device)
        )
        self.model.paths, self.model.frames, self.model.latents = paths, frames, latents
        recon_frames = self.model.extract_latents(
            num_steps=opt.steps,
            save_path=experiment_dir,
            batch_size=opt.batch_size,
            timesteps_to_save=timesteps_to_save,
            inversion_prompt=opt.inversion_prompt,
        )

        os.mkdir(os.path.join(experiment_dir, "invert_frames"))
        for i, frame in enumerate(recon_frames):
            T.ToPILImage()(frame).save(
                os.path.join(experiment_dir, f"invert_frames", f"{i:05d}.png")
            )
        print("Inversion completed!")

        output_dir = os.path.join(experiment_dir, "output_dir")
        os.makedirs(output_dir)
        batch_size = 8
        if not max_frames % batch_size == 0:
            # make n_frames divisible by batch_size
            max_frames = max_frames - (max_frames % batch_size)
        print("Number of frames for editing: ", max_frames)
        tokenflow_pnp_config = {
            "seed": seed,
            "device": self.device,
            "inversion_prompt": inversion_prompt,
            "output_path": output_dir,
            "data_path": opt.data_path,
            "latents_path": os.path.join(experiment_dir, "latents"),
            "n_inversion_steps": num_inversion_steps,
            "n_frames": max_frames,
            "sd_version": self.sd_version,
            "guidance_scale": guidance_scale,
            "n_timesteps": num_diffusion_steps,
            "prompt": diffusion_prompt,
            "negative_prompt": negative_diffusion_prompt,
            "batch_size": batch_size,
            "pnp_attn_t": 0.5,
            "pnp_f_t": 0.8,
        }

        self.editor.config_editor(tokenflow_pnp_config)
        edited_frames = self.editor.edit_video()

        out_path = "/tmp/out.mp4"
        save_video(edited_frames, out_path, fps=fps)
        print("Done!")

        return Path(out_path)


def save_intput_video_frames(video_path, video_frames_dir, img_size=(512, 512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith(".mov"):
        video = T.functional.rotate(video, -90)
    os.makedirs(video_frames_dir, exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size), resample=Image.Resampling.LANCZOS)
        image_resized.save(f"{video_frames_dir}/{ind}.png")


def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    return timesteps, num_inference_steps - t_start


def get_data(frames_path, n_frames, device):
    # load frames
    total_frames = len(os.listdir(frames_path))
    max_frames = min(total_frames, n_frames)
    paths = [f"{frames_path}/%05d.png" % i for i in range(max_frames)]
    frames = [Image.open(path).convert("RGB") for path in paths]
    if frames[0].size[0] == frames[0].size[1]:
        frames = [
            frame.resize((512, 512), resample=Image.Resampling.LANCZOS)
            for frame in frames
        ]
    frames = (
        torch.stack([T.ToTensor()(frame) for frame in frames])
        .to(torch.float16)
        .to(device)
    )
    return paths, frames, max_frames


class Preprocess(nn.Module):
    def __init__(self, device, opt, hf_key=None):
        super().__init__()
        self.device = device
        self.sd_version = opt.sd_version
        self.use_depth = False

        print(f"[INFO] loading stable diffusion...")
        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5" or self.sd_version == "ControlNet":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "depth":
            model_key = "stabilityai/stable-diffusion-2-depth"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )
        self.model_key = model_key
        # Create model
        self.vae = AutoencoderKL.from_pretrained(
            model_key,
            subfolder="vae",
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=opt.cache_dir,
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_key,
            subfolder="tokenizer",
            cache_dir=opt.cache_dir,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key,
            subfolder="text_encoder",
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=opt.cache_dir,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key,
            subfolder="unet",
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=opt.cache_dir,
        ).to(self.device)

        if self.sd_version == "ControlNet":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
                cache_dir=opt.cache_dir,
            ).to(self.device)
            control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                cache_dir=opt.cache_dir,
            ).to(self.device)
            self.unet = control_pipe.unet
            self.controlnet = control_pipe.controlnet
            self.canny_cond = self.get_canny_cond()
        elif self.sd_version == "depth":
            self.depth_maps = self.prepare_depth_maps()
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            cache_dir=opt.cache_dir,
        )
        print(f"[INFO] loaded stable diffusion!")

    @torch.no_grad()
    def prepare_depth_maps(self, model_type="DPT_Large", device="cuda"):
        depth_maps = []
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        for i in range(len(self.paths)):
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            latent_h = img.shape[0] // 8
            latent_w = img.shape[1] // 8

            input_batch = transform(img).to(device)
            prediction = midas(input_batch)

            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(latent_h, latent_w),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_maps.append(depth_map)

        return torch.cat(depth_maps).to(self.device).to(torch.float16)

    @torch.no_grad()
    def get_canny_cond(self):
        canny_cond = []
        for image in self.frames.cpu().permute(0, 2, 3, 1):
            image = np.uint8(np.array(255 * image))
            low_threshold = 100
            high_threshold = 200

            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = torch.from_numpy((image.astype(np.float32) / 255.0))
            canny_cond.append(image)
        canny_cond = (
            torch.stack(canny_cond)
            .permute(0, 3, 1, 2)
            .to(self.device)
            .to(torch.float16)
        )
        return canny_cond

    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1,
            return_dict=False,
        )

        # apply the denoising network
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        decoded = []
        batch_size = 8
        for b in range(0, latents.shape[0], batch_size):
            latents_batch = 1 / 0.18215 * latents[b : b + batch_size]
            imgs = self.vae.decode(latents_batch).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            decoded.append(imgs)
        return torch.cat(decoded)

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=10, deterministic=True):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i : i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def ddim_inversion(
        self,
        cond,
        latent_frames,
        save_path,
        batch_size,
        save_latents=True,
        timesteps_to_save=None,
    ):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = (
            timesteps_to_save if timesteps_to_save is not None else timesteps
        )
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b : b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                if self.sd_version == "depth":
                    depth_maps = torch.cat([self.depth_maps[b : b + batch_size]])
                    model_input = torch.cat([x_batch, depth_maps], dim=1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0
                    else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = (
                    self.unet(model_input, t, encoder_hidden_states=cond_batch).sample
                    if self.sd_version != "ControlNet"
                    else self.controlnet_pred(
                        x_batch,
                        t,
                        cond_batch,
                        torch.cat([self.canny_cond[b : b + batch_size]]),
                    )
                )
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b : b + batch_size] = mu * pred_x0 + sigma * eps

            if save_latents and t in timesteps_to_save:
                torch.save(
                    latent_frames,
                    os.path.join(save_path, "latents", f"noisy_latents_{t}.pt"),
                )
        torch.save(
            latent_frames, os.path.join(save_path, "latents", f"noisy_latents_{t}.pt")
        )
        return latent_frames

    @torch.no_grad()
    def ddim_sample(self, x, cond, batch_size):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b : b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)

                if self.sd_version == "depth":
                    depth_maps = torch.cat([self.depth_maps[b : b + batch_size]])
                    model_input = torch.cat([x_batch, depth_maps], dim=1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = (
                    self.unet(model_input, t, encoder_hidden_states=cond_batch).sample
                    if self.sd_version != "ControlNet"
                    else self.controlnet_pred(
                        x_batch,
                        t,
                        cond_batch,
                        torch.cat([self.canny_cond[b : b + batch_size]]),
                    )
                )

                pred_x0 = (x_batch - sigma * eps) / mu
                x[b : b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
        return x

    @torch.no_grad()
    def extract_latents(
        self, num_steps, save_path, batch_size, timesteps_to_save, inversion_prompt=""
    ):
        self.scheduler.set_timesteps(num_steps)
        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        latent_frames = self.latents

        inverted_x = self.ddim_inversion(
            cond,
            latent_frames,
            save_path,
            batch_size=batch_size,
            save_latents=True,
            timesteps_to_save=timesteps_to_save,
        )
        latent_reconstruction = self.ddim_sample(
            inverted_x, cond, batch_size=batch_size
        )

        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction


class TokenFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        sd_version = config["sd_version"]
        self.sd_version = sd_version
        if sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == "depth":
            model_key = "stabilityai/stable-diffusion-2-depth"
        else:
            raise ValueError(f"Stable-diffusion version {sd_version} not supported.")

        # Create SD models
        print("Loading SD model")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=torch.float16, cache_dir=config["cache_dir"]
        ).to("cuda")

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", cache_dir=config["cache_dir"]
        )
        print("SD model loaded")

    def config_editor(self, config):
        self.config = config
        self.scheduler.set_timesteps(self.config["n_timesteps"], device=self.device)

        # data
        self.latents_path = self.config["latents_path"]
        # load frames
        self.paths, self.frames, self.latents, self.eps = self.get_data()
        if self.sd_version == "depth":
            self.depth_maps = self.prepare_depth_maps()

        self.text_embeds = self.get_text_embeds(
            config["prompt"], config["negative_prompt"]
        )
        pnp_inversion_prompt = config["inversion_prompt"]
        self.pnp_guidance_embeds = self.get_text_embeds(
            pnp_inversion_prompt, pnp_inversion_prompt
        ).chunk(2)[0]

    @torch.no_grad()
    def prepare_depth_maps(self, model_type="DPT_Large", device="cuda"):
        depth_maps = []
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        for i in range(len(self.paths)):
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            latent_h = img.shape[0] // 8
            latent_w = img.shape[1] // 8

            input_batch = transform(img).to(device)
            prediction = midas(input_batch)

            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(latent_h, latent_w),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_maps.append(depth_map)

        return torch.cat(depth_maps).to(torch.float16).to(self.device)

    def get_pnp_inversion_prompt(self):
        inv_prompts_path = os.path.join(
            str(Path(self.latents_path).parent), "inversion_prompt.txt"
        )
        # read inversion prompt
        with open(inv_prompts_path, "r") as f:
            inv_prompt = f.read()
        return inv_prompt

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat(
            [uncond_embeddings] * batch_size + [text_embeddings] * batch_size
        )
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i : i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i : i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_data(self):
        # load frames
        paths = [
            os.path.join(self.config["data_path"], "%05d.png" % idx)
            for idx in range(self.config["n_frames"])
        ]
        frames = [
            Image.open(paths[idx]).convert("RGB")
            for idx in range(self.config["n_frames"])
        ]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [
                frame.resize((512, 512), resample=Image.Resampling.LANCZOS)
                for frame in frames
            ]
        frames = (
            torch.stack([T.ToTensor()(frame) for frame in frames])
            .to(torch.float16)
            .to(self.device)
        )
        # save_video(frames, f'{self.config["output_path"]}/input_fps10.mp4', fps=10)
        # save_video(frames, f'{self.config["output_path"]}/input_fps20.mp4', fps=20)
        # save_video(frames, f'{self.config["output_path"]}/input_fps30.mp4', fps=30)
        # encode to latents
        latents = (
            self.encode_imgs(frames, deterministic=True)
            .to(torch.float16)
            .to(self.device)
        )
        # get noise
        eps = (
            self.get_ddim_eps(latents, range(self.config["n_frames"]))
            .to(torch.float16)
            .to(self.device)
        )
        return paths, frames, latents, eps

    def get_ddim_eps(self, latent, indices):
        noisest = max(
            [
                int(x.split("_")[-1].split(".")[0])
                for x in glob.glob(
                    os.path.join(self.latents_path, f"noisy_latents_*.pt")
                )
            ]
        )
        latents_path = os.path.join(self.latents_path, f"noisy_latents_{noisest}.pt")
        noisy_latent = torch.load(latents_path)[indices].to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T**0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.latents_path)[indices]
        latent_model_input = torch.cat([source_latents] + ([x] * 2))
        if self.sd_version == "depth":
            latent_model_input = torch.cat(
                [latent_model_input, torch.cat([self.depth_maps[indices]] * 3)], dim=1
            )

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat(
            [
                self.pnp_guidance_embeds.repeat(len(indices), 1, 1),
                torch.repeat_interleave(self.text_embeds, len(indices), dim=0),
            ]
        )

        # apply the denoising network
        noise_pred = self.unet(
            latent_model_input, t, encoder_hidden_states=text_embed_input
        )["sample"]

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (
            noise_pred_cond - noise_pred_uncond
        )

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        return denoised_latent

    @torch.autocast(dtype=torch.float16, device_type="cuda")
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config["batch_size"]
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x) // batch_size,)) + torch.arange(
            0, len(x), batch_size
        )

        register_pivotal(self, True)
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        register_pivotal(self, False)
        for i, b in enumerate(range(0, len(x), batch_size)):
            register_batch_idx(self, i)
            denoised_latents.append(
                self.denoise_step(x[b : b + batch_size], t, indices[b : b + batch_size])
            )
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = (
            self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        )
        self.conv_injection_timesteps = (
            self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        )
        register_extended_attention_pnp(self, self.qk_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)
        set_tokenflow(self.unet)

    # def save_vae_recon(self):
    #     os.makedirs(f'{self.config["output_path"]}/vae_recon', exist_ok=True)
    #     decoded = self.decode_latents(self.latents)
    #     for i in range(len(decoded)):
    #         T.ToPILImage()(decoded[i]).save(f'{self.config["output_path"]}/vae_recon/%05d.png' % i)
    #     save_video(decoded, f'{self.config["output_path"]}/vae_recon_10.mp4', fps=10)
    #     save_video(decoded, f'{self.config["output_path"]}/vae_recon_20.mp4', fps=20)
    #     save_video(decoded, f'{self.config["output_path"]}/vae_recon_30.mp4', fps=30)

    def edit_video(self):
        # os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        # self.save_vae_recon()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        noisy_latents = self.scheduler.add_noise(
            self.latents, self.eps, self.scheduler.timesteps[0]
        )
        edited_frames = self.sample_loop(
            noisy_latents, torch.arange(self.config["n_frames"])
        )
        return edited_frames

    def sample_loop(self, x, indices):
        # os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            x = self.batched_denoise_step(x, t, indices)

        decoded_latents = self.decode_latents(x)
        # for i in range(len(decoded_latents)):
        #     T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % i)

        return decoded_latents
