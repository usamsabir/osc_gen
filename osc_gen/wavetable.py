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
from __future__ import print_function
import numpy as np

from osc_gen import wavfile
from osc_gen import dsp
from osc_gen import sig


class WaveTable(object):
    """ An n-slot wavetable """

    def __init__(self, waves=None, num_waves=16, wave_len=128):
        """
        Init

        @param waves sequence : A sequence of numpy arrays containing wave
            data to form the wavetable
        """

        self.num_waves = num_waves
        self.wave_len = wave_len

        if waves is None:
            self.waves = []
        else:
            self.waves = waves

    def clear(self):
        """ Clear the wavetable so that all slots contain zero """

        self.waves = []

    def get_wave_at_index(self, index):
        """
        Get the wave at a specific slot index

        @param index int : The slot index to get the wave from

        @returns np.ndarray : Wave at given index
        """

        if index >= len(self.waves):
            return np.zeros(self.wave_len)

        return self.waves[index]

    def get_waves(self):
        """ Get all of the waves in the table """

        for i in range(self.num_waves):
            yield self.get_wave_at_index(i)

    def from_wav(self, filename, sig_gen=None, resynthesize=False):
        """
        Populate the wavetable from a wav file by filling all slots with
        cycles from a wav file.

        @param filename str : Wav file name.
        @param sig_gen SigGen : SigGen to use for regenerating the signal.
        @param resynthesize : If True, the signal is resynthesised using the
            harmonic series of the original signal - works best on signals with
            a low findamental frequency (< 200 Hz). If False, n evenly spaced
            single cycles are extracted from the input (default False).

        @returns WaveTable : self, populated by content from the wav file
        settings as this one
        """

        data, fs = wavfile.read(filename, with_sample_rate=True)

        if sig_gen is None:
            sig_gen = sig.SigGen()

        if resynthesize:

            num_sections = self.num_waves

            while True:
                data = data[:data.size - (data.size % num_sections)]
                sections = np.split(data, num_sections)
                try:
                    self.waves = [dsp.resynthesize(s, sig_gen) for s in sections]
                    break
                except dsp.NotEnoughSamplesError as exc:
                    num_sections -= 1
                    if num_sections <= 0:
                        raise exc

            if num_sections < self.num_waves:
                self.waves = sig.morph(self.waves, self.num_waves)

        else:
            cycles = dsp.slice_cycles(data, self.num_waves, fs)
            self.waves = [sig_gen.arb(c) for c in cycles]

        return self

    def morph_with(self, other):
        """ Morph waves with contents of another wavetable """

        waves = [None for _ in range(self.num_waves)]

        for i in range(self.num_waves):
            wav_a = self.get_wave_at_index(i)
            wav_b = other.get_wave_at_index(i)
            # interpolate wav_b to the same length as a
            if len(wav_a) != len(wav_b):
                wav_b = sig.SigGen(num_points=len(wav_a)).arb(wav_b)
            waves[i] = sig.morph([wav_a, wav_b], 3)[1]

        return WaveTable(waves)
