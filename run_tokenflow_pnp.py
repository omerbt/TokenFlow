import glob
import os
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from tokenflow_utils import *
from util import save_video, seed_everything

# suppress partial model loading warning
logging.set_verbosity_error()

VAE_BATCH_SIZE = 10


class TokenFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        
        sd_version = config["sd_version"]
        self.sd_version = sd_version
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        # data
        self.latents_path = self.get_latents_path()
        # load frames
        self.paths, self.frames, self.latents, self.eps = self.get_data()
        if self.sd_version == 'depth':
            self.depth_maps = self.prepare_depth_maps()

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        pnp_inversion_prompt = self.get_pnp_inversion_prompt()
        self.pnp_guidance_embeds = self.get_text_embeds(pnp_inversion_prompt, pnp_inversion_prompt).chunk(2)[0]
    
    @torch.no_grad()   
    def prepare_depth_maps(self, model_type='DPT_Large', device='cuda'):
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
        inv_prompts_path = os.path.join(str(Path(self.latents_path).parent), 'inversion_prompt.txt')
        # read inversion prompt
        with open(inv_prompts_path, 'r') as f:
            inv_prompt = f.read()
        return inv_prompt

    def get_latents_path(self):
        latents_path = os.path.join(config["latents_path"], f'sd_{config["sd_version"]}',
                             Path(config["data_path"]).stem, f'steps_{config["n_inversion_steps"]}')
        latents_path = [x for x in glob.glob(f'{latents_path}/*') if '.' not in Path(x).name]
        n_frames = [int([x for x in latents_path[i].split('/') if 'nframes' in x][0].split('_')[1]) for i in range(len(latents_path))]
        latents_path = latents_path[np.argmax(n_frames)]
        self.config["n_frames"] = min(max(n_frames), config["n_frames"])
        if self.config["n_frames"] % self.config["batch_size"] != 0:
            # make n_frames divisible by batch_size
            self.config["n_frames"] = self.config["n_frames"] - (self.config["n_frames"] % self.config["batch_size"])
        print("Number of frames: ", self.config["n_frames"])
        return os.path.join(latents_path, 'latents')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    
    def get_data(self):
        # load frames
        paths = [os.path.join(config["data_path"], "%05d.jpg" % idx) for idx in
                               range(self.config["n_frames"])]
        if not os.path.exists(paths[0]):
            paths = [os.path.join(config["data_path"], "%05d.png" % idx) for idx in
                                   range(self.config["n_frames"])]
        frames = [Image.open(paths[idx]).convert('RGB') for idx in range(self.config["n_frames"])]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        save_video(frames, f'{self.config["output_path"]}/input_fps10.mp4', fps=10)
        save_video(frames, f'{self.config["output_path"]}/input_fps20.mp4', fps=20)
        save_video(frames, f'{self.config["output_path"]}/input_fps30.mp4', fps=30)
        # encode to latents
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        # get noise
        eps = self.get_ddim_eps(latents, range(self.config["n_frames"])).to(torch.float16).to(self.device)
        return paths, frames, latents, eps

    def get_ddim_eps(self, latent, indices):
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(self.latents_path, f'noisy_latents_*.pt'))])
        latents_path = os.path.join(self.latents_path, f'noisy_latents_{noisest}.pt')
        noisy_latent = torch.load(latents_path)[indices].to(self.device)
        alpha_prod_T = self.scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    @torch.no_grad()
    def denoise_step(self, x, t, indices):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.latents_path)[indices]
        latent_model_input = torch.cat([source_latents] + ([x] * 2))
        if self.sd_version == 'depth':
            latent_model_input = torch.cat([latent_model_input, torch.cat([self.depth_maps[indices]] * 3)], dim=1)

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds.repeat(len(indices), 1, 1),
                                      torch.repeat_interleave(self.text_embeds, len(indices), dim=0)])

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent
    
    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, x, t, indices):
        batch_size = self.config["batch_size"]
        denoised_latents = []
        pivotal_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0,len(x),batch_size) 
            
        register_pivotal(self, True)
        self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
        register_pivotal(self, False)
        for i, b in enumerate(range(0, len(x), batch_size)):
            register_batch_idx(self, i)
            denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
        denoised_latents = torch.cat(denoised_latents)
        return denoised_latents

    def init_method(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_extended_attention_pnp(self, self.qk_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)
        set_tokenflow(self.unet)

    def save_vae_recon(self):
        os.makedirs(f'{self.config["output_path"]}/vae_recon', exist_ok=True)
        decoded = self.decode_latents(self.latents)
        for i in range(len(decoded)):
            T.ToPILImage()(decoded[i]).save(f'{self.config["output_path"]}/vae_recon/%05d.png' % i)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_10.mp4', fps=10)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_20.mp4', fps=20)
        save_video(decoded, f'{self.config["output_path"]}/vae_recon_30.mp4', fps=30)

    def edit_video(self):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        self.save_vae_recon()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        noisy_latents = self.scheduler.add_noise(self.latents, self.eps, self.scheduler.timesteps[0])
        edited_frames = self.sample_loop(noisy_latents, torch.arange(self.config["n_frames"]))
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_10.mp4')
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_20.mp4', fps=20)
        save_video(edited_frames, f'{self.config["output_path"]}/tokenflow_PnP_fps_30.mp4', fps=30)
        print('Done!')

    def sample_loop(self, x, indices):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.batched_denoise_step(x, t, indices)
        
        decoded_latents = self.decode_latents(x)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % i)

        return decoded_latents


def run(config):
    seed_everything(config["seed"])
    print(config)
    editor = TokenFlow(config)
    editor.edit_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_pnp.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["output_path"] = os.path.join(config["output_path"] + f'_pnp_SD_{config["sd_version"]}',
                                             Path(config["data_path"]).stem,
                                             config["prompt"][:240],
                                             f'attn_{config["pnp_attn_t"]}_f_{config["pnp_f_t"]}',
                                             f'batch_size_{str(config["batch_size"])}',
                                             str(config["n_timesteps"]),
    )
    os.makedirs(config["output_path"], exist_ok=True)
    assert os.path.exists(config["data_path"]), "Data path does not exist"
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    run(config)
