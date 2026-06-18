# TanGO: Training-Free 3D Editing via Tangent-space Guidance and Optimization

<a href='<PROJECT_PAGE_URL>'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='<ARXIV_URL>'><img src='https://img.shields.io/badge/Technique-Report-red'></a>

This repo provides the official implementation of **TanGO**, a **training-free** framework for 3D editing on VecSet-based flow models. Instead of applying a single global update, TanGO performs **adaptive per-token steering in the tangent space** of the generative dynamics: it amplifies guidance for tokens in regions to be edited while attenuating it for tokens to be preserved, enabling **mask-free**, localized, identity-preserving edits.

![teaser](./assets/teaser.png)

At each ODE step we form the source/target velocity difference and modulate it with a per-token gain derived from a von Mises–Fisher directional discrepancy:

```
Δv_i      = v_tar,i − v_src,i                  # raw velocity difference  (Eq. 2)
cos θ_i   = ⟨v̂_tar,i , v̂_src,i⟩                # directional agreement     (Eq. 7)
d_i       = 1 − cos θ_i                         # token-wise demand         (Eq. 10)
λ_eff(t)  = λ · ( c / mean_i d_i )              # mean-gain normalization   (Eq. 12)
u_i(t)    = λ_eff(t) · d_i · Δv_i               # per-token control input   (Eq. 13)
```

The entire method lives in [`src/tango.py`](src/tango.py) — see `TanGOEdit._tango_control`.

## Installation

TanGO builds upon [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), which serves as the base flow model. We recommend using CUDA 12.4 (as suggested in the official Hunyuan3D instructions) or CUDA 12.1.

```bash
# Create a conda environment
conda create -n tango python=3.10
```

* CUDA=12.4

```bash
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
--index-url https://download.pytorch.org/whl/cu124
# Install dependencies
pip install -r requirements_cuda124.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

* CUDA=12.1 (also support CUDA=12.2)

```bash
# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
--index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cuda121.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

The Hunyuan3D-2.1 weights are downloaded automatically from the Hugging Face Hub on first run (`tencent/Hunyuan3D-2.1`).

## Usage

Run TanGO with default settings (paper defaults λ=5.0, c=0.2, n_max=41). More examples can be found in `./examples`.

```bash
python3 src/tango.py
```

To edit a specific example:

```bash
python3 src/tango.py \
  --src_mesh examples/dragon/src.glb \
  --src_img  examples/dragon/src.png \
  --tar_img  examples/dragon/edited.png \
  --output   examples/dragon/tango.glb \
  --lam 5.0 --c 0.2 --n_max 41
```

## Prepare Editing Data

Given a `source shape` and `editing prompt`, we first construct the editing conditions, including the `source image` and `target image`. Then, we organize the inputs and perform 3D editing.

* Step1: Rendering the `source image` from the `source shape`.
  Use the **Render Multiview Images** script in [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). Then select one suitable rendering as the `source image`.
* Step2: Construct the `target image` using a 2D editing model: Apply a 2D editing model (e.g., [Nano Banana](https://aistudio.google.com/models/gemini-2-5-flash-image)) to edit the `source image` according to the given `editing prompt`, producing the `target image`.
* Step3: Organize the inputs and perform 3D editing. Place all required inputs in a folder and run the 3D editing pipeline. The process requires the following files:
  * `src.glb`: the `source shape`
  * `src.png`: the `source image`
  * `edited.png`: the `target image`

You can find example setups in `./examples`.

## Notes

- If you have questions or find bugs, feel free to open an issue or email the first author (<EMAIL>).
- If you encounter `EGL: cannot open shared object file: No such file or directory` error during rendering mesh, try to install following packages: `sudo apt-get install libegl1-mesa libgl1-mesa-glx`.

## Acknowledgements

Our repo is built on top of several awesome projects and works, including [FlowEdit](https://github.com/fallenshock/FlowEdit), [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) and [AnchorFlow](https://github.com/ZhenglinZhou/AnchorFlow).

## Cite

If you find TanGO useful for your research and applications, please cite us using this BibTex:

```bibtex
@inproceedings{tango2026,
  title={TanGO: Training-Free 3D Editing via Tangent-space Guidance and Optimization},
  author={<AUTHORS>},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026},
}
```
