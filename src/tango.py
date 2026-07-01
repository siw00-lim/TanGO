
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Resolve paths relative to the repo root (parent of this src/ directory) so the
# script can be launched from anywhere, e.g. `python3 src/tango.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                              # for torchvision_fix
sys.path.insert(0, os.path.join(_ROOT, 'hy3dshape'))   # Hunyuan3D base flow model

# ShapeVAE in Hunyuan3D
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
from hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.pipelines import export_to_trimesh

# ShapeDiT in Hunyuan3D
from hy3dshape.pipelines import retrieve_timesteps, Hunyuan3DDiTPipeline

try:
    from torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


class Hunyuan3DVAE:
    def __init__(
            self,
            model_path='tencent/Hunyuan3D-2.1',
            enable_flashvdm=False,
    ):
        self.vae = ShapeVAE.from_pretrained(
            model_path=model_path,
            use_safetensors=False,
            variant='fp16',
        )
        if enable_flashvdm:
            self.vae.enable_flashvdm_decoder(
                enabled=True,
                adaptive_kv_selection=True,
                topk_mode='mean',
                mc_algo='mc'
            )

        self.loader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920,
        )

    def encode(self, mesh_path):
        surface = self.loader(mesh_path).to('cuda', dtype=torch.float16)
        print(surface.shape)
        latents = self.vae.encode(surface)
        return latents

    @torch.no_grad()
    def decode(self, latents, save_path=None):
        latents = self.vae.decode(latents)
        mesh = self.vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=1.01,
            mc_level=0.0,
            num_chunks=20000,
            octree_resolution=256,
            mc_algo='mc',
            enable_pbar=True
        )
        mesh = export_to_trimesh(mesh)[0]
        if save_path is not None:
            mesh.export(save_path)
            print(f"Successfully saved result to {save_path}")
        return mesh


class TanGOEdit(Hunyuan3DDiTPipeline):
    """Per-token tangent-space steering on top of a VecSet flow model."""

    def make_condition_lat(self, image, guidance_scale, mask=None):
        do_classifier_free_guidance = guidance_scale >= 0 and not (
                hasattr(self.model, 'guidance_embed') and
                self.model.guidance_embed is True
        )
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        return cond

    def _get_guidance(self, guidance_scale, batch_size, device, dtype):
        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
                self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
        return guidance

    def _tango_control(self, t: torch.Tensor, zt_src: torch.Tensor, zt_tar: torch.Tensor):
        """Compute the per-token control update u_i(t) at a single ODE step.

        Returns lambda_eff * (1 - cos theta_i) * dv_i, with dv_i = v_tar,i - v_src,i.
        """
        _zt_src = torch.cat([zt_src] * 2)
        _zt_tar = torch.cat([zt_tar] * 2)

        B = _zt_src.shape[0]
        t_batch = (t if isinstance(t, torch.Tensor) else torch.tensor(t, device=_zt_src.device))
        t_batch = t_batch.to(device=_zt_src.device, dtype=torch.float32).repeat(B)
        t_batch = t_batch / self.scheduler.config.num_train_timesteps

        # Classifier-free guidance for source- and target-conditioned velocities.
        vt_src_pred = self.model(_zt_src, t_batch, self.src_cond_lat, guidance=self.src_guidance)
        vt_src_cond, vt_src_uncond = vt_src_pred.chunk(2)
        vt_src = vt_src_uncond + self.src_guidance_scale * (vt_src_cond - vt_src_uncond)

        vt_tar_pred = self.model(_zt_tar, t_batch, self.tar_cond_lat, guidance=self.tar_guidance)
        vt_tar_cond, vt_tar_uncond = vt_tar_pred.chunk(2)
        vt_tar = vt_tar_uncond + self.tar_guidance_scale * (vt_tar_cond - vt_tar_uncond)

        raw_dv = vt_tar - vt_src  # dv_i = v_tar,i - v_src,i        (Eq. 2)

        # Token-wise directional discrepancy -> adaptive per-token gain.
        cos_sim = F.cosine_similarity(vt_tar, vt_src, dim=-1)       # cos theta_i   (Eq. 7)
        d_i = (1.0 - cos_sim).clamp(min=0).unsqueeze(-1)           # demand d_i    (Eq. 10)

        # Mean-gain normalization keeps overall guidance energy constant over steps.
        mean_d = d_i.mean().clamp(min=1e-6)
        lambda_eff = self.lam * (self.c / mean_d)                  # lambda_eff    (Eq. 12)
        output = lambda_eff * d_i * raw_dv                        # u_i(t)        (Eq. 13)

        if self.traj_log_enabled:
            step_index = self.scheduler.index_for_timestep(t)
            self.traj_log.append({
                'step': int(step_index),
                'sigma': float(self.scheduler.sigmas[step_index]),
                'w_div': d_i.detach().squeeze(-1).squeeze(0).float().cpu(),  # d_i = 1 - cos θ_i
                'u_norm': output.detach().norm(dim=-1).squeeze(0).float().cpu(),
                'cos_sim': cos_sim.detach().squeeze(0).float().cpu(),
            })
        return output

    def _propagate_for_timestep(self, zt_inv: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        B = zt_inv.shape[0]
        t_batch = (t if isinstance(t, torch.Tensor) else torch.tensor(t, device=zt_inv.device))
        t_batch = t_batch.to(device=zt_inv.device, dtype=torch.float32).repeat(B)

        u_avg = 0
        for _ in range(self.n_avg):
            fwd_noise = self.fwd_noise if self.fixed_noise else torch.randn_like(self.x_src)
            zt_src = self.scheduler.scale_noise(self.x_src, t_batch, fwd_noise)
            # FlowEdit coupling: edited trajectory shares the source noise offset.
            zt_tar = zt_inv + zt_src - self.x_src
            u_avg += self._tango_control(t, zt_src, zt_tar)
        u = u_avg / self.n_avg

        zt_inv = zt_inv.to(torch.float32) + dt * u
        return zt_inv.to(u.dtype)

    @torch.no_grad()
    def denoise(self, x_src, src_cond_img, tar_cond_img, edit_kwargs):
        # ── editing hyper-parameters ──
        self.T_steps = edit_kwargs.get('T_steps', 50)
        self.src_guidance_scale = edit_kwargs.get('src_guidance_scale', 3.5)
        self.tar_guidance_scale = edit_kwargs.get('tar_guidance_scale', 7.5)
        self.n_avg = edit_kwargs.get('n_avg', 1)
        self.n_max = edit_kwargs.get('n_max', 41)
        self.fixed_noise = edit_kwargs.get('fixed_noise', True)
        self.lam = edit_kwargs.get('lam', 5.0)                   # lambda (global scale)
        self.c = edit_kwargs.get('c', 0.2)                       # target mean gain
        self.traj_log_enabled = edit_kwargs.get('traj_log', False)
        if self.traj_log_enabled:
            self.traj_log = []

        self.x_src = x_src
        self.src_cond_img, self.tar_cond_img = src_cond_img, tar_cond_img
        device, dtype = self.device, self.dtype
        batch_size = x_src.shape[0]

        # condition latents (source / target images)
        self.src_guidance = self._get_guidance(self.src_guidance_scale, batch_size, device, dtype)
        self.tar_guidance = self._get_guidance(self.tar_guidance_scale, batch_size, device, dtype)
        self.src_cond_lat = self.make_condition_lat(self.src_cond_img, guidance_scale=self.src_guidance_scale)
        self.tar_cond_lat = self.make_condition_lat(self.tar_cond_img, guidance_scale=self.tar_guidance_scale)

        # timesteps / sigmas
        sigmas = np.linspace(0, 1, self.T_steps)
        timesteps, _ = retrieve_timesteps(self.scheduler, self.T_steps, device, sigmas=sigmas)

        # fixed forward noise sampled once and shared across all steps
        self.fwd_noise = torch.randn_like(self.x_src)

        # editing loop: integrate from n_max down to n_min
        zt_inv = self.x_src.clone()
        zt_inv_list = []
        start_index = max(0, len(timesteps) - self.n_max)
        for i in tqdm(range(start_index, len(timesteps) - 1), desc="TanGO editing"):
            t = timesteps[i]
            t_i = t / self.scheduler.config.num_train_timesteps
            t_im1 = timesteps[i + 1] / self.scheduler.config.num_train_timesteps
            dt = t_im1 - t_i
            zt_inv = self._propagate_for_timestep(zt_inv, t, dt)
            zt_inv_list.append(zt_inv.clone())

        return zt_inv, zt_inv_list


def load_image(image_path, save_path=None):
    if save_path and os.path.exists(save_path):
        image = Image.open(save_path).convert('RGBA')
        print(f"Successfully loaded image from {save_path}")
    else:
        image = Image.open(image_path).convert("RGBA")
        rembg = BackgroundRemover()
        image = rembg(image)
        if save_path is not None:
            image.save(save_path)
            print(f"Successfully saved preprocessed image to {save_path}")
    return image


def edit_3d_model(
        src_img_path,
        tar_img_path,
        src_mesh,
        n_max=41,
        src_guidance_scale=3.5,
        tar_guidance_scale=7.5,
        lam=5.0,
        c=0.2,
        fixed_noise=True,
        traj_log=False,
        output_path='examples/dragon/tango.glb',
        vis_trajectory=False,
):
    """Run TanGO editing on a single (source mesh, source image, target image)."""
    src_img = load_image(src_img_path, src_img_path.replace(".png", "_rm_bg.png"))
    tar_img = load_image(tar_img_path, tar_img_path.replace(".png", "_rm_bg.png"))

    latents = hunyuan_vae.encode(src_mesh)

    edit_latents, edit_trajectory = hunyuan_3dedit.denoise(latents, src_img, tar_img, {
        'T_steps': 50,
        'n_max': n_max,
        'src_guidance_scale': src_guidance_scale,
        'tar_guidance_scale': tar_guidance_scale,
        'lam': lam,
        'c': c,
        'fixed_noise': fixed_noise,
        'traj_log': traj_log,
    })

    if vis_trajectory:
        output_path = Path(output_path)
        out_dir = output_path.parent / output_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, latent in enumerate(edit_trajectory):
            hunyuan_vae.decode(latent, save_path=str(out_dir / f"{i}.glb"))
    else:
        hunyuan_vae.decode(edit_latents, save_path=output_path)

    return latents, edit_trajectory


def set_seed(seed=42):
    import os, random, numpy as np, torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TanGO: training-free 3D editing")
    parser.add_argument('--src_mesh', default='examples/dragon/src.glb',
                        help='source 3D mesh (.glb)')
    parser.add_argument('--src_img', default='examples/dragon/src.png',
                        help='rendered image of the source object')
    parser.add_argument('--tar_img', default='examples/dragon/edited.png',
                        help='target (edited) image describing the desired edit')
    parser.add_argument('--output', default='examples/dragon/tango.glb', help='output .glb path')
    parser.add_argument('--model_path', default='tencent/Hunyuan3D-2.1', help='base flow model')
    parser.add_argument('--n_max', type=int, default=41, help='number of editing steps (from n_max down to n_min)')
    parser.add_argument('--src_guidance_scale', type=float, default=3.5)
    parser.add_argument('--tar_guidance_scale', type=float, default=7.5)
    parser.add_argument('--lam', type=float, default=5.0, help='global guidance scale lambda')
    parser.add_argument('--c', type=float, default=0.2, help='target mean gain (mean-gain normalization)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis_trajectory', action='store_true', help='decode and save every step')
    args = parser.parse_args()

    global hunyuan_vae, hunyuan_3dedit
    hunyuan_vae = Hunyuan3DVAE(model_path=args.model_path)
    hunyuan_3dedit = TanGOEdit.from_pretrained(model_path=args.model_path)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    edit_3d_model(
        src_img_path=args.src_img,
        tar_img_path=args.tar_img,
        src_mesh=args.src_mesh,
        n_max=args.n_max,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        lam=args.lam, c=args.c,
        output_path=args.output,
        vis_trajectory=args.vis_trajectory,
    )
    print(f"[OK] {args.output}")


if __name__ == '__main__':
    main()
