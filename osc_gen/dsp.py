#!/usr/bin/env python3
"""
Copyright 2019 Harvey Ormston

This file is part of osc_gen.

    osc_gen is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    osc_gen is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with osc_gen.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import division

import cmath
import math
from copy import deepcopy

import numpy as np


class NotEnoughSamplesError(Exception):
    """Not Enough Samples"""


def normalize(inp):
    """Normalize a signal to the range +/- 1

    @param inp seq : A sequence of samples
    """

    dc_bias = (np.amax(inp) + np.amin(inp)) / 2
    inp -= dc_bias
    amp = np.amax(np.absolute(inp))

    if amp > 0:
        inp /= amp

    return inp


def mix(inp_a, inp_b, amount=0.5):
    """Mix two signals together.

    @param inp_a np.ndarray : first input
    @param inp_b np.ndarray : seconds input
    @param amount float : mix amount, 0 outputs only inp_a, 1 outputs only
        inp_b, values between 0 and 1 output a propotional mix of the two.
    """

    amount = np.clip(amount, 0, 1)

    return normalize((inp_a * (1 - amount) + inp_b * amount))


def clip(inp, amount, bias=0):
    """Hard-clip a signal

    @param inp seq : A sequence of samples
    @param amount number : Amount of clipping
    @param bias number : Pre-distortion DC bias
    """

    gain = 1 + amount

    inp += bias
    inp *= gain
    np.clip(inp, -1.0, 1.0, out=inp)

    return normalize(inp)


def tube(inp, amount, bias=0):
    """Tube saturate a signal

    @param inp seq : A sequence of samples
    @param amount number : Amount of distortion
    @param bias number : Pre-distortion DC bias
    """

    gain = 1 + amount
    inp += bias
    inp *= gain
    for i, val in enumerate(inp):
        inp[i] = math.exp(-np.logaddexp(0, -val))

    return normalize(inp)


def fold(inp, amount, bias=0):
    """Perform wave folding

    @param inp seq : A sequence of samples
    @param amount number : Amount of distortion
    @param bias number : Pre-distortion DC bias
    """

    gain = 1 + amount
    inp += bias
    inp *= gain
    while (np.amax(np.abs(inp))) > 1:
        for i, val in enumerate(inp):
            if val > 1:
                inp[i] = 2 - val
            if val < -1:
                inp[i] = -2 - val

    return normalize(inp)


def shape(inp, amount=1, bias=0, power=3):
    """Perform polynomial waveshaping

    @param inp seq : A sequence of samples
    @param amount number : Amount of shaping
        (1: maximum shaping, 0: no shaping)
    @param bias number : Pre-distortion DC bias
    @param power number : Polynomial power
    """

    biased = inp + bias

    # make another copy to apply polynomial shaping to the biased input
    shaped = np.empty_like(biased)
    # shape positive and negative halves of the signal symmetrically
    shaped[biased >= 0] = np.power(biased[biased >= 0], power) * amount
    shaped[biased < 0] = -np.power(-biased[biased < 0], power) * amount
    # de-bais
    shaped -= bias
    normalize(shaped)

    inp *= 1 - amount
    inp += shaped * amount

    return normalize(inp)


def slew(inp, rate, inv=False):
    """Apply slew or overhoot to a signal. Slew smooths steep transients in
    the signal while overshoot results in a sharper transient with
    ringing.

    @param rate float : Slew rate, between 0 and 1
    @param inp seq : A sequence of samples
    @param inv bool : If True, overshoot will be applied. if False,
                      slew will be applied. (default=False).
    """

    if inv:
        beta = 1 - rate
    else:
        beta = rate

    alpha = 1 - beta

    # the output is the middle cycle of 3, shifted slightly to account for
    # filter run-in
    start = inp.size - 2
    end = (2 * inp.size) - 2

    tiled_inp = np.tile(inp, 3)

    for idx, sample in enumerate(tiled_inp[1:]):
        tiled_inp[idx] = (sample * beta) + (tiled_inp[idx - 1] * alpha)

    return normalize(tiled_inp[start:end])


def downsample(inp, factor):
    """Reduce the effective sample rate of a signal, resulting in aliasing.

    @param inp seq : A sequence of samples
    @param factor int : Downsampling factor
    """

    if factor < 1:
        raise ValueError(f"Downsampling factor ({factor}) cannot be < 1")

    if factor == 1:
        return inp

    # the aliasing is deliberate!
    for i, val in enumerate(inp):
        if i % factor == 0:
            last = val
        else:
            inp[i] = last

    return normalize(inp)


def quantize(inp, depth):
    """Reduce the bit depth of a signal.

    @param inp seq : A sequence of samples
    @param depth number : New bit depth in bits
    """

    scale = 2**depth - 1

    for i, val in enumerate(inp):
        if val > 0:
            inp[i] = np.ceil(val * scale) / scale
        elif val < 0:
            inp[i] = np.floor(val * scale) / scale

    return normalize(inp)


def fundamental(inp, fs):
    """Find the fundamental frequency in Hz of a given input"""

    window = np.hamming(inp.size)
    sig = np.fft.fft(inp * window)
    freqs = np.fft.fftfreq(sig.size)
    i = np.argmax(np.abs(sig))

    return np.abs(freqs[i] * fs)


def harmonic_series(inp):
    """Find the harmonic series of a periodic input"""

    fft_mult = min(64, inp.size // 501)
    fft_mult = max(fft_mult, 1)
    fft_len = 501 * fft_mult

    # if the input has insufficient data, loop it a number of times
    while inp.size < fft_len:
        inp = np.tile(inp, 2)

    # produce symmetrical, windowed fft
    idx1 = int(np.floor((fft_len + 1) / 2))
    idx2 = int(np.floor(fft_len / 2))
    windowed = inp[:fft_len] * np.hamming(fft_len)
    fft_half = 1024 * fft_mult
    buf = np.zeros(fft_half)
    buf[:idx1] = windowed[idx2:]
    buf[fft_half - idx2 :] = windowed[:idx2]
    fft = np.fft.fft(buf)[: fft_half // 2]

    # peak amplitude assumed to be fundamental frequency
    i_fund = np.argmax(np.abs(fft))
    i_fund = max(i_fund, 4)

    # get fft components from only the harmonics, harmonics are picked by
    # taking the value with the highest amplitude around each harmonic
    # frequency
    start = i_fund // 4
    harmonics = np.array(
        [
            fft[i - start : i + start][np.abs(fft[i - start : i + start]).argmax()]
            for i in range(i_fund, fft_half // 2, i_fund)
        ]
    )

    # normalize magnitude and phase
    hs_amp = np.abs(harmonics)
    hs_ang = np.angle(harmonics)
    harmonics = hs_amp * np.exp(1j * (hs_ang - hs_ang[0])) / hs_amp[0]

    return harmonics


def slice_cycles(inp, n, fs):
    """Extact n single-cycle slices from a signal"""

    def nearest(arr, val):
        """find the nearest value in an array to a given value"""
        return arr[np.argmin(np.abs(arr - val))]

    zero_crossings = np.where(np.diff(np.sign(inp)) > 0)[0] + 1

    if not zero_crossings.size:
        raise ValueError("No zero crossings found.")

    freq = fundamental(inp, fs)
    samples_per_cycle = fs / freq
    end = len(inp) - samples_per_cycle

    slots = np.linspace(0, end, n)
    slots = np.around(slots).astype(int)
    slots = np.unique([nearest(zero_crossings, slot) for slot in slots])

    return [inp[x : x + int(samples_per_cycle)] for x in slots]


def resynthesize(inp, sig_gen):
    """
    Resynthesize a signal from its harmonic series

    @param sig_gen SigGen : SigGen to use for regenerating the signal.
    """

    sine_gen = deepcopy(sig_gen)
    max_harmonic = sig_gen.num_points // 2
    harmonics = harmonic_series(inp)
    outp = np.zeros(sine_gen.num_points)

    for i, harmonic in enumerate(harmonics):

        sine_gen.harmonic = i
        sine_gen.amp = np.abs(harmonic)
        sine_gen.phase = np.angle(harmonic)

        outp += sine_gen.sin()

        if i >= max_harmonic:
            break

    return normalize(outp)

def fft(x):
    N = len(x)

    # Base case for recursion
    if N <= 1:
        return x

    # Recursive case
    even = fft(x[0::2])
    odd = fft(x[1::2])

    # Combine
    T = [cmath.exp(-2j * math.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def ifft(X):
    N = len(X)

    # Compute the inverse FFT
    x = fft([X[k].conjugate() for k in range(N)])

    # Scale the result
    return [x[k].conjugate() / N for k in range(N)]

def signal_mixer(signal1: list[list[float]], signal2: list[list[float]], operation: str) -> list[list[float]]:
    """
    Performs operations on two input signals represented as 2D lists.
    Args:
        signal1: First input signal as a 2D list.
        signal2: Second input signal as a 2D list.
        operation: The operation to perform ('addition', 'subtraction', 'inverse').
    Returns:
        A new 2D list representing the result of the operation.
    Raises:
        ValueError: If invalid input or operation is provided.
    """
    rows1 = len(signal1)
    cols1 = len(signal1[0]) if rows1 > 0 else 0
    rows2 = len(signal2)
    cols2 = len(signal2[0]) if rows2 > 0 else 0

    if operation == 'addition':
        if rows1 != rows2 or cols1 != cols2:
            raise ValueError("Invalid")
        return [[signal1[i][j] + signal2[i][j] for j in range(cols1)] for i in range(rows1)]

    elif operation == 'subtraction':
        if rows1 != rows2 or cols1 != cols2:
            raise ValueError("Invalid")
        return [[signal1[i][j] - signal2[i][j] for j in range(cols1)] for i in range(rows1)]

    elif operation == 'inverse':
        if rows1 != cols1:
            raise ValueError("Invalid")

        n = rows1
        # Create an augmented matrix [A|I]
        augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(signal1)]

        # Gaussian elimination
        for i in range(n):
            if augmented[i][i] == 0:
                raise ValueError("Invalid")

            # Make the diagonal element 1
            factor = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= factor

            # Make other elements in the column 0
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]

        # Extract the inverse matrix
        inverse = [row[n:] for row in augmented]
        return inverse

    else:
        raise ValueError("Invalid")
