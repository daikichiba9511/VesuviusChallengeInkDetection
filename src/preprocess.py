from __future__ import annotations

import cupy as cp
import numpy as np

xp = cp
delta_lookup = {
    "xx": xp.array([[1, -2, 1]], dtype=float),
    "yy": xp.array([[1], [-2], [1]], dtype=float),
    "xy": xp.array([[1, -1], [-1, 1]], dtype=float),
}


def operate_derivative(img_shape, pair):
    assert len(img_shape) == 2
    delta = delta_lookup[pair]
    fft = xp.fft.fftn(delta, img_shape)
    return fft * xp.conj(fft)


def soft_threshold(vector, threshold):
    """Soft thresholding function.

    Args:
        vector: (r, n)
        threshold: float

    Returns:
        (r, n)
    """
    return xp.sign(vector) * xp.maximum(xp.abs(vector) - threshold, 0)


def back_diff(input_image, dim):
    """Backward difference along a given dimension.

    Args:
        input_image: (r, n)
        dim: 0 or 1

    Returns:
        (r, n)
    """
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r + 1, n + 1), dtype=float)
    temp2 = xp.zeros((r + 1, n + 1), dtype=float)

    temp1[position[0] : size[0], position[1] : size[1]] = input_image
    temp2[position[0] : size[0], position[1] : size[1]] = input_image

    size[dim] += 1
    position[dim] += 1
    temp2[position[0] : size[0], position[1] : size[1]] = input_image
    temp1 -= temp2
    size[dim] -= 1
    return temp1[0 : size[0], 0 : size[1]]


def forward_diff(input_image, dim):
    """Forward difference along a given dimension.

    Args:
        input_image: (r, n)
        dim: 0 or 1

    Returns:
        (r, n)w

    """
    assert dim in (0, 1)
    r, n = xp.shape(input_image)
    size = xp.array((r, n))
    position = xp.zeros(2, dtype=int)
    temp1 = xp.zeros((r + 1, n + 1), dtype=float)
    temp2 = xp.zeros((r + 1, n + 1), dtype=float)

    size[dim] += 1
    position[dim] += 1

    temp1[position[0] : size[0], position[1] : size[1]] = input_image
    temp2[position[0] : size[0], position[1] : size[1]] = input_image

    size[dim] -= 1
    temp2[0 : size[0], 0 : size[1]] = input_image
    temp1 -= temp2
    size[dim] += 1
    return -temp1[position[0] : size[0], position[1] : size[1]]


def iter_deriv(input_image, b, scale, mu, dim1, dim2):
    g = back_diff(forward_diff(input_image, dim1), dim2)
    d = soft_threshold(g + b, 1 / mu)
    b = b + (g - d)
    L = scale * back_diff(forward_diff(d - b, dim2), dim1)
    return L, b


def iter_xx(*args):
    return iter_deriv(*args, dim1=1, dim2=1)


def iter_yy(*args):
    return iter_deriv(*args, dim1=0, dim2=0)


def iter_xy(*args):
    return iter_deriv(*args, dim1=0, dim2=1)


def iter_sparse(input_image, bsparse, scale, mu):
    d = soft_threshold(input_image + bsparse, 1 / mu)
    bsparse = bsparse + (input_image - d)
    Lsparse = scale * (d - bsparse)
    return Lsparse, bsparse


def denoise_image(
    input_image: np.ndarray | cp.ndarray,
    iter_num: int = 100,
    fidelity: int = 150,
    sparsity_scale: int = 10,
    continuity_scale: float = 0.5,
    mu: int = 1,
) -> np.ndarray | cp.ndarray:
    """画像のノイズ除去

    Args:
        input_image: ノイズ除去する画像
        iter_num: 繰り返し回数
        fidelity: ノイズ除去の信頼度
        sparsity_scale: スパース性の強さ
        continuity_scale: 連続性の強さ
        mu: ラグランジュ乗数

    Returns:
        ノイズ除去後の画像
    """
    image_size = xp.shape(input_image)
    # print("Initialize denoising")
    norm_array = (
        operate_derivative(image_size, "xx")
        + operate_derivative(image_size, "yy")
        + 2 * operate_derivative(image_size, "xy")
    )
    norm_array += (fidelity / mu) + sparsity_scale**2
    b_arrays = {
        "xx": xp.zeros(image_size, dtype=float),
        "yy": xp.zeros(image_size, dtype=float),
        "xy": xp.zeros(image_size, dtype=float),
        "L1": xp.zeros(image_size, dtype=float),
    }
    g_update = xp.multiply(fidelity / mu, input_image)
    for i in range(iter_num):
        # print(f"Starting iteration {i+1}")
        g_update = xp.fft.fftn(g_update)
        if i == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
        g_update = xp.multiply((fidelity / mu), input_image)

        # print("XX update")
        L, b_arrays["xx"] = iter_xx(g, b_arrays["xx"], continuity_scale, mu)
        g_update += L

        # print("YY update")
        L, b_arrays["yy"] = iter_yy(g, b_arrays["yy"], continuity_scale, mu)
        g_update += L

        # print("XY update")
        L, b_arrays["xy"] = iter_xy(g, b_arrays["xy"], 2 * continuity_scale, mu)
        g_update += L

        # print("L1 update")
        L, b_arrays["L1"] = iter_sparse(g, b_arrays["L1"], sparsity_scale, mu)
        g_update += L

    g_update = xp.fft.fftn(g_update)
    g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real

    g[g < 0] = 0
    g -= g.min()
    g /= g.max()
    return g
