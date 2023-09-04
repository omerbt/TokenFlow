from typing import Type
import torch
import os

from util import isinstance_str, batch_cosine_sim

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

def register_extended_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject unconditional
                q[n_frames:2 * n_frames] = q[:n_frames]
                k[n_frames:2 * n_frames] = k[:n_frames]
                # inject conditional
                q[2 * n_frames:] = q[:n_frames]
                k[2 * n_frames:] = k[:n_frames]

            k_source = k[:n_frames]
            k_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_source = v[:n_frames]
            v_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames:2 * n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)


            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            out_source_all = []
            out_uncond_all = []
            out_cond_all = []
            
            single_batch = n_frames<=12
            b = n_frames if single_batch else 1

            for frame in range(0, n_frames, b):
                out_source = []
                out_uncond = []
                out_cond = []
                for j in range(h):
                    sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * self.scale

                    out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                    out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                    out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))

                out_source = torch.cat(out_source, dim=0)
                out_uncond = torch.cat(out_uncond, dim=0) 
                out_cond = torch.cat(out_cond, dim=0) 
                if single_batch:
                    out_source = out_source.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_uncond = out_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_cond = out_cond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_source_all.append(out_source)
                out_uncond_all.append(out_uncond)
                out_cond_all.append(out_cond)
            
            out_source = torch.cat(out_source_all, dim=0)
            out_uncond = torch.cat(out_uncond_all, dim=0)
            out_cond = torch.cat(out_cond_all, dim=0)
                
            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            k_source = k[:n_frames]
            k_uncond = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_source = v[:n_frames]
            v_uncond = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames: 2*n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            out_source = []
            out_uncond = []
            out_cond = []

            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):
                sim_source_b = torch.bmm(q_src[:, j], k_src[:, j].transpose(-1, -2)) * self.scale
                sim_uncond_b = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                sim_cond = torch.bmm(q_cond[:, j], k_cond[:, j].transpose(-1, -2)) * self.scale

                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[:, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[:, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[:, j]))

            out_source = torch.cat(out_source, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_cond = torch.cat(out_cond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)

def make_tokenflow_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class TokenFlowBlock(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            mid_idx = n_frames // 2
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            else:
                idx1 = []
                idx2 = [] 
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)
                
                sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
                                        self.pivot_hidden_states[0][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                    idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                    idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                    idx2 = idx2.squeeze(1)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.pivotal_pass:
                # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                self.attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )
                # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                self.kf_attn_output = self.attn_output 
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,
                                   batch_idxs]  # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.pivotal_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))

                    s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                    # distance from the pivot
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))

                attn_output = attn_output.reshape(
                        batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

    return TokenFlowBlock


def set_tokenflow(
        model: torch.nn.Module):
    """
    Sets the tokenflow attention blocks in a model.
    """

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tokenflow_block_fn = make_tokenflow_attention_block 
            module.__class__ = make_tokenflow_block_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model
