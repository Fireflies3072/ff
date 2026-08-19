# ff

Fireflies' personal utility library for Python, focusing on computer vision and generative deep learning.

## Installation

Install directly with `pip`:

```bash
pip install git+https://github.com/Fireflies3072/ff.git
```

Install directly with `uv`:

```bash
uv add git+https://github.com/Fireflies3072/ff.git
```

To install the latest version from source:

```bash
git clone https://github.com/Fireflies3072/ff.git
cd ff
pip install .
```

## Notes

On Linux, import order is not an issue. On Windows, importing `torch` **before** `pyarrow>=24.0` can cause a very long hang. Older versions of `pyarrow` are unaffected.

If you need both, import libraries that pull in `pyarrow` first (for example `datasets`), then import `torch`. Because `ff.nn` uses both, import `ff.nn` as early as possible in your project so the safe order is applied automatically.

## Features

### Computer Vision (`ff.cv`)
- **Image Compression**: `jpeg_compress` for simulating JPEG artifacts.
- **Smart Resize**: `resize_cover` for resizing images to a target size while maintaining aspect ratio and cropping from the center.

### General Utilities (`ff.utils`)
- **String Manipulation**: `to_snake_case` for converting text to safe filenames or variable names.

### Neural Networks (`ff.nn`)
A comprehensive toolkit for building and training generative models:

- **Architecture**: 
    - Diffusion Transformer (DiT) blocks for 1D, 2D, and 3D data.
    - Sinusoidal and Rotary Position Embeddings (RoPE).
    - Self-attention and residual blocks.
- **Diffusion & Flow Matching**:
    - Implementation of DiT and UNet for diffusion.
    - Multiple samplers and schedulers for inference.
    - Support for Classifier-Free Guidance (CFG).
    - Flow matching samplers.
- **Training Utilities**:
    - Exponential Moving Average (EMA) for model weights.
    - Perceptual Loss (LPIPS) and VAE processing.
    - Functional utilities for tensor operations.
- **Datasets**:
    - Specialized dataset loaders for image generation tasks.

## License

MIT
