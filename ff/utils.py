import re
import numpy as np

def to_snake_case(text: str) -> str:
    """Convert arbitrary text to a safe lowercase, snake_case string."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_")

def random_smooth_noise(shape: tuple[int, ...], cutoff_ratio: float = 0.05,
                        decay_power: float = 2.0, seed: int | None = None) -> np.ndarray:
    """Generate a seamless smooth noise field with an N-D real inverse FFT.

    Args:
        shape: Output array shape, any number of dimensions,
            e.g. (1024,), (512, 512), (32, 64, 64), (8, 16, 32, 32).
        cutoff_ratio: Low-frequency cutoff ratio.
            (Recommended: 0.01 ~ 0.2). (Allowed: 0 ~ 0.5).
            Smaller values produce wider, gentler transitions.
        decay_power: Frequency-domain power-spectrum decay exponent
            (Recommended: 1.0 ~ 2.5). (Allowed: 0 ~ infinity).
            Larger values look softer, like Brownian noise.
        seed: Random seed.

    Returns:
        Numpy array of the given shape, normalized to [0, 1].
    """
    
    if not shape:
        raise ValueError("shape must have at least 1 dimension")

    rng = np.random.default_rng(seed)
    ndim = len(shape)
    freq_shape = (*shape[:-1], shape[-1] // 2 + 1)
    # Complex random frequency values
    freq = rng.standard_normal(freq_shape) + 1j * rng.standard_normal(freq_shape)
    # Set zero frequency to zero
    origin = (0,) * ndim
    freq[origin] = 0

    # Frequency components along each axis
    freqs = []
    for axis, size in enumerate(shape):
        k = np.fft.rfftfreq(size) if axis == ndim - 1 else np.fft.fftfreq(size)
        freqs.append(k.reshape((1,) * axis + (-1,) + (1,) * (ndim - axis - 1)))
    radius = np.sqrt(sum(k**2 for k in freqs))
    # Remove zero frequency
    radius[origin] = 1

    # Apply cutoff and decay
    freq *= (radius <= cutoff_ratio) / radius**decay_power
    # Inverse Fourier transform and normalize to [0, 1]
    field = np.fft.irfftn(freq, s=shape)
    return (field - field.min()) / (field.max() - field.min() + 1e-8)
