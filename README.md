# TanGO: Training-Free 3D Editing via Tangent-space Guidance and Optimization

<a href='<PROJECT_PAGE_URL>'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='<ARXIV_URL>'><img src='https://img.shields.io/badge/Technique-Report-red'></a>

This repo provides the official implementation of **TanGO**, a **training-free** framework for 3D editing on VecSet-based flow models. Instead of applying a single global update, TanGO performs **adaptive per-token steering in the tangent space** of the generative dynamics: it amplifies guidance for tokens in regions to be edited while attenuating it for tokens to be preserved, enabling **mask-free**, localized, identity-preserving edits.

![teaser](./assets/teaser.png)

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

### Quick start (a simple recipe for a single edit)

Given a `source shape` and an `editing prompt`, we first construct the editing conditions — a `source image` and a `target image` — then organize the inputs and run 3D editing.

* Step1: Render the `source image` from the `source shape`.
  Use the **Render Multiview Images** script in [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). Then select one suitable rendering as the `source image`.
* Step2: Construct the `target image` using a 2D editing model: Apply a 2D editing model (e.g., [Nano Banana](https://aistudio.google.com/models/gemini-2-5-flash-image)) to edit the `source image` according to the given `editing prompt`, producing the `target image`.
* Step3: Organize the inputs and perform 3D editing. Place all required inputs in a folder and run the 3D editing pipeline. The process requires the following files:
  * `src.glb`: the `source shape`
  * `src.png`: the `source image`
  * `edited.png`: the `target image`

You can find example setups in `./examples`.

### TanGOEdit benchmark

The quick-start recipe above is meant for trying out a single edit. The actual evaluation benchmark used in the paper, **TanGOEdit**, is built with the multi-stage, VLM-assisted pipeline detailed in the supplementary material (Sec. C). It contains **100 curated editing instances** roughly balanced across **five edit categories** — *Add, Replace, Action Change, Style Change, Remove* — and is constructed as follows:

* **Step1 — Source asset collection.** Collect diverse 3D assets with clear object identity and visually interpretable geometry; inspect rendered views and keep assets whose main structure and editable regions are clearly visible.
* **Step2 — Editing instruction construction.** For each asset, prompt a vision-language model ([Qwen-VL-2.5](https://github.com/QwenLM/Qwen2.5-VL), following [Nano3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)'s instruction strategy) to assign one of the five categories and generate a single, specific, identity-preserving, and visually grounded editing instruction.
* **Step3 — Quality control and selection.** Remove or rewrite ambiguous, unverifiable, or overly destructive instructions, then select 100 high-quality samples with a roughly balanced category distribution.

Please refer to the supplementary material for the full prompt template and details.

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
