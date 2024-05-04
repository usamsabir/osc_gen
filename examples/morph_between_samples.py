#!/usr/bin/env python3

"""
This example creates a wave table by interpolating between two or more user-defined
(single-cycle) samples. Result is written to an output WAV file.
"""

import argparse
import sys

import soundfile as sf

from osc_gen import sig, visualize, wavetable


def parse_command_line():
    """Parse command line arguments"""

    # Create parser
    parser = argparse.ArgumentParser(
        description="Interpolate 2 or more single-cycle samples to wave table, "
        "and write result to WAV file."
    )

    # Add arguments
    parser.add_argument("slots", action="store", type=int, help="number of slots")
    parser.add_argument("file_out", action="store", type=str, help="output WAV file")
    parser.add_argument(
        "samples_in", action="store", type=str, nargs="+", help="input samples"
    )

    # Parse arguments
    args = parser.parse_args()

    return args


def main():
    """main function"""

    # Parse command line arguments
    args = parse_command_line()

    slots = args.slots
    file_out = args.file_out
    samples = args.samples_in

    if len(samples) < 2:
        sys.stderr.write("Error: number of input samples must be 2 or more\n")
        sys.exit()

    # Establish wave length from first sample
    sample_data, _sample_rate = sf.read(samples[0])
    wave_length = len(sample_data)

    # Create signal generator instance
    sg = sig.SigGen(num_points=wave_length)

    # Create wave table instance
    wt = wavetable.WaveTable(slots)

    # In the following block we create a list that contains the data from
    # all input samples

    waves_in = []

    for sample in samples:
        # Read sample data
        try:
            sample_data, _sample_rate = sf.read(sample)
        except:  # pylint: disable=bare-except
            sys.stderr.write("Error: cannot read file " + sample + "\n")
            sys.exit()
        # Create arbitrary wave instance from sample data
        wave_in = sg.arb(sample_data)
        # Append to list of input waves
        waves_in.append(wave_in)

    # Fill all wave table slots by interpolating between input waves
    wt.waves = sig.morph(waves_in, slots)

    # Plot resulting wave table
    visualize.plot_wavetable(wt)

    # Write to wav file
    wt.to_wav(file_out)


if __name__ == "__main__":
    main()
