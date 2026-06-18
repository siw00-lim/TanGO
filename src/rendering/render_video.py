"""
Render 360° turntable video from a GLB mesh.
Usage:
  python src/rendering/render_video.py --mesh_path foo.glb --save_path foo.mp4
  python src/rendering/render_video.py --mesh_path foo.glb --save_path foo.mp4 --n_frames 72 --fps 30
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import pyrender
import imageio


def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def render_turntable_video(
    mesh_path, save_path,
    n_frames=72, fps=30,
    elev_deg=20, radius=2.2,
    w=512, h=512,
    bg=(255, 255, 255, 255),
):
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh.remove_unreferenced_vertices()
    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / max(mesh.extents.max(), 1e-6)
    mesh.apply_scale(scale * 1.2)

    scene = pyrender.Scene(bg_color=bg, ambient_light=np.array([0.15, 0.15, 0.15, 1.0]))

    has_texture = hasattr(mesh.visual, 'kind') and mesh.visual.kind == 'texture'
    has_vertex_colors = (
        (hasattr(mesh.visual, 'kind') and mesh.visual.kind == 'vertex') or
        (hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and
         not np.all(mesh.visual.vertex_colors[:, :3] == mesh.visual.vertex_colors[0, :3]))
    )

    if has_texture:
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    elif has_vertex_colors:
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.82, 0.82, 0.82, 1.0),
            metallicFactor=0.0, roughnessFactor=1.0
        )
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=material)
    scene.add(pm)

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    cam_node = scene.add(cam, pose=np.eye(4))

    for eye_l, intensity in [((2.5, 2.0, 1.5), 1.0), ((-2.5, 1.5, -0.5), 0.6), ((0.0, 2.5, -2.5), 0.8)]:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        scene.add(light, pose=look_at(eye_l))

    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    elev_rad = np.deg2rad(elev_deg)
    frames = []
    for i in range(n_frames):
        az = 2 * np.pi * i / n_frames
        eye = np.array([
            radius * np.cos(elev_rad) * np.cos(az),
            radius * np.sin(elev_rad),
            radius * np.cos(elev_rad) * np.sin(az),
        ])
        scene.set_pose(cam_node, pose=look_at(eye))
        color, _ = r.render(scene)
        frames.append(color[..., :3])

    r.delete()

    save_path = str(save_path)

    # Write frames as temp PNGs, then ffmpeg → mp4
    import tempfile, subprocess, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f'{i:04d}.png'))
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(tmpdir, '%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23', '-preset', 'fast',
            save_path
        ], check=True, capture_output=True)
    finally:
        shutil.rmtree(tmpdir)
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    render_turntable_video(args.mesh_path, args.save_path,
                           n_frames=args.n_frames, fps=args.fps,
                           w=args.resolution, h=args.resolution)
    print(f"[OK] {args.save_path}")
