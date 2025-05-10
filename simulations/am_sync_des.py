import math
import cmath

import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt


SR = 480000 // 32
FIX_MAX = (1 << 15) - 1  # corresponds to 8
FIX_ONE = (1 << 12) - 1  # corresponds to 1
FIX_PI = round(FIX_ONE * np.pi)
FIX_ERR_SCALE = round(FIX_ONE * math.pi / 8)
FIX_PHI_SCALE = round(FIX_ONE * 8 / (2 * math.pi))


def rectangular_2_phase(i, q):
    if i == 0 and q == 0:
        return 0

    absi = i if i > 0 else -i
    angle = 0
    if q >= 0:
        r = ((q - absi) << 13) / (q + absi)
        angle = 8192 - r
    else:
        r = ((q + absi) << 13) / (absi - q)
        angle = (3 * 8192) - r

    angle = int(angle)
    if i < 0:
        return -angle
    else:
        return angle


class PLL:
    def __init__(self, loop_bw, freq_min, freq_max):
        self.loop_bw = 2 * math.pi * (loop_bw / SR)
        self.freq_min = 2 * math.pi * (freq_min / SR)
        self.freq_max = 2 * math.pi * (freq_max / SR)

        damping = math.sqrt(2) / 2
        self.loop_bw = self.loop_bw / (damping + 1 / (4 * damping))
        denom = 1 + 2 * damping * self.loop_bw + self.loop_bw * self.loop_bw
        self.alpha = (4 * damping * self.loop_bw) / denom
        self.beta = (4 * self.loop_bw * self.loop_bw) / denom

        self.phi_locked = 0.0
        self.freq_locked = (self.freq_min + self.freq_max) / 2.0

    def process(self, i, q):
        out = complex(math.cos(self.phi_locked), math.sin(self.phi_locked))
        err = cmath.phase(complex(i, q) * complex(out.real, -out.imag))

        self.freq_locked = self.freq_locked + self.beta * err
        self.phi_locked = self.phi_locked + self.freq_locked + self.alpha * err

        self.freq_locked = (
            (self.freq_locked > self.freq_max) and self.freq_max or self.freq_locked
        )
        self.freq_locked = (
            (self.freq_locked < self.freq_min) and self.freq_min or self.freq_locked
        )

        self.phi_locked = (
            (self.phi_locked > 2 * math.pi)
            and (self.phi_locked - 2 * math.pi)
            or self.phi_locked
        )
        self.phi_locked = (
            (self.phi_locked < -2 * math.pi)
            and (self.phi_locked + 2 * math.pi)
            or self.phi_locked
        )

        return out.real, out.imag, err


class PLLFixed:
    def __init__(self, loop_bw, freq_min, freq_max):

        self.loop_bw = 2 * math.pi * (loop_bw / SR)
        self.freq_min = 2 * math.pi * (freq_min / SR)
        self.freq_max = 2 * math.pi * (freq_max / SR)

        damping = math.sqrt(2) / 2
        self.loop_bw = self.loop_bw / (damping + 1 / (4 * damping))
        denom = 1 + 2 * damping * self.loop_bw + self.loop_bw * self.loop_bw

        self.alpha = (4 * damping * self.loop_bw) / denom
        self.beta = (4 * self.loop_bw * self.loop_bw) / denom

        print(self.alpha, self.beta, self.freq_min, self.freq_max)

        self.freq_min = round(self.freq_min * FIX_ONE)
        self.freq_max = round(self.freq_max * FIX_ONE)

        self.alpha = round(self.alpha * FIX_ONE)
        self.beta = round(self.beta * FIX_ONE)

        self.phi_locked = 0
        self.freq_locked = round((self.freq_min + self.freq_max) / 2)

        scaling_factor = (1 << 15) - 1
        self.sin_table = []
        for idx in range(2048):
            self.sin_table.append(
                round(math.sin(2.0 * math.pi * idx / 2048.0) * scaling_factor)
            )

    def process(self, i, q):

        idx = (self.phi_locked * FIX_PHI_SCALE) >> 12
        if idx < 0:
            idx = FIX_MAX + 1 + idx

        out_i = self.sin_table[((idx // 16) + 512) & 0x7FF]
        out_q = self.sin_table[(idx // 16) & 0x7FF]

        tmp_i = (i * out_i + q * out_q) >> 15
        tmp_q = (-i * out_q + q * out_i) >> 15

        err = (-rectangular_2_phase(tmp_i, tmp_q) * FIX_ERR_SCALE) >> 12

        self.freq_locked = self.freq_locked + ((self.beta * err) >> 12)
        self.phi_locked = (
            self.phi_locked + self.freq_locked + ((self.alpha * err) >> 12)
        )

        self.freq_locked = (
            self.freq_max if (self.freq_locked > self.freq_max) else self.freq_locked
        )

        self.freq_locked = (
            self.freq_min if (self.freq_locked < self.freq_min) else self.freq_locked
        )

        self.phi_locked = (
            self.phi_locked - (2 * FIX_PI)
            if (self.phi_locked > (2 * FIX_PI))
            else self.phi_locked
        )

        self.phi_locked = (
            self.phi_locked + (2 * FIX_PI)
            if (self.phi_locked < -(2 * FIX_PI))
            else self.phi_locked
        )

        return -out_q, out_i, err


def floating_sim(loop_bw, freq_min, freq_max, time, input):
    pll = PLL(loop_bw, freq_min, freq_max)
    print(pll.alpha, pll.beta, pll.freq_min, pll.freq_max)

    output = []
    error = []
    for s in input:
        out_i, out_q, err = pll.process(np.real(s), np.imag(s))
        output.append(complex(out_i, out_q))
        error.append(err)

    output = np.array(output)
    input1 = np.real(input_a)

    plt.plot(time, input1, label="input real")
    plt.plot(time, np.real(output), label="output real")
    plt.plot(time, error, label="phase error")

    plt.grid(True)
    plt.legend()
    plt.show()


def fixed_sim(loop_bw, freq_min, freq_max, time, input):
    pll = PLLFixed(loop_bw, freq_min, freq_max)

    print(f"#define AMSYNC_ALPHA ({pll.alpha})")
    print(f"#define AMSYNC_BETA ({pll.beta})")
    print(f"#define AMSYNC_F_MIN ({pll.freq_min})")
    print(f"#define AMSYNC_F_MAX ({pll.freq_max})")
    print(f"#define AMSYNC_PI ({FIX_PI})")
    print(f"#define AMSYNC_ONE ({FIX_ONE})")
    print(f"#define AMSYNC_MAX ({FIX_MAX})")
    print(f"#define AMSYNC_ERR_SCALE ({FIX_ERR_SCALE})")
    print(f"#define AMSYNC_PHI_SCALE ({FIX_PHI_SCALE})")

    print(pll.alpha, pll.beta, pll.freq_min, pll.freq_max)

    output = []
    error = []
    for s in input:
        out_i, out_q, err = pll.process(
            round(np.real(s) * FIX_ONE), round(np.imag(s) * FIX_ONE)
        )
        output.append(complex(out_i, out_q))
        error.append(err)

    output = np.array(output)
    input1 = np.round(np.real(input_a) * FIX_ONE)

    plt.plot(time, input1, label="input real")
    plt.plot(time, np.real(output), label="output real")
    plt.plot(time, error, label="phase error")

    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    time = np.linspace(0, 0.1, SR // 10)
    input = (np.sin(2 * np.pi * 50 * time + 0.8) * 0.95) + np.random.random(
        len(time)
    ) * 0.01
    input_a = sig.hilbert(input)

    floating_sim(100, -3000, 3000, time, input_a)
    fixed_sim(100, -3000, 3000, time, input_a)
