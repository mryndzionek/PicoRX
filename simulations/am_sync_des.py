import math
import cmath

import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt
from cffi import FFI


SR = 480000 // 32
BASE_BITS = 15
BASE_MAX = (1 << BASE_BITS) - 1

FIX_ONE_BITS = 18
FIX_MAX = (1 << (FIX_ONE_BITS + 3)) - 1  # times 8 to accommodate 2*pi
FIX_ONE = (1 << FIX_ONE_BITS) - 1
FIX_PI = round(FIX_ONE * np.pi)
FIX_ERR_SCALE = round(FIX_PI / BASE_MAX)
FIX_PHI_SCALE = round(16 / (((BASE_MAX) / (2 * FIX_PI))))


SRC = """
// PLL loop bandwidth: 50Hz
#define AMSYNC_NUM_TAPS (3)
#define AMSYNC_B0 (13241)
#define AMSYNC_B1 (-26352)
#define AMSYNC_B2 (13114)
#define AMSYNC_A0 (262143)
#define AMSYNC_A1 (-524286)
#define AMSYNC_A2 (262143)
#define AMSYNC_PI (823547)
#define AMSYNC_ONE (262143)
#define AMSYNC_MAX (2097151)
#define AMSYNC_ERR_SCALE (25)
#define AMSYNC_PHI_SCALE (804)
#define AMSYNC_FRACTION_BITS (18)
#define AMSYNC_BASE_FRACTION_BITS (15)

static int16_t sin_table[2048];

void initialise_luts(void)
{
  //pre-generate sin/cos lookup tables
  float scaling_factor = (1 << 15) - 1;
  for(uint16_t idx=0; idx<2048; idx++)
  {
    sin_table[idx] = roundf(sinf(2.0*M_PI*idx/2048.0) * scaling_factor);
  }
}

int16_t rectangular_2_phase(int16_t i, int16_t q)
{
   //handle condition where phase is unknown
   if(i==0 && q==0) return 0;

   const int16_t absi=i>0?i:-i;
   int16_t angle=0;
   if (q>=0)
   {
      //scale r so that it lies in range -8192 to 8192
      const int16_t r = ((int32_t)(q - absi) << 13) / (q + absi);
      angle = 8192 - r;
   }
   else
   {
      //scale r so that it lies in range -8192 to 8192
      const int16_t r = ((int32_t)(q + absi) << 13) / (absi - q);
      angle = (3 * 8192) - r;
   }

   //angle lies in range -32768 to 32767
   if (i < 0) return(-angle);     // negate if in quad III or IV
   else return(angle);
}

void amsync_demod(int16_t *i, int16_t *q, int64_t *err) {
  static int64_t phi_locked = 0;
  static int64_t x1 = 0;
  static int64_t y1 = 0;
  static int64_t x2 = 0;
  static int64_t y2 = 0;
  static int64_t y0_err = 0;

  size_t idx = (phi_locked / AMSYNC_PHI_SCALE);

  if (phi_locked < 0) {
    idx = 2048 + idx;
  }

  // VCO
  const int32_t vco_i = sin_table[(idx + 512u) & 0x7ffu];
  const int32_t vco_q = sin_table[idx & 0x7ffu];

  // Phase Detector
  const int16_t synced_i = (*i * vco_i + *q * vco_q) >> AMSYNC_BASE_FRACTION_BITS;
  const int16_t synced_q =
      (-*i * vco_q + *q * vco_i) >> AMSYNC_BASE_FRACTION_BITS;

  *err =
      ((int64_t)rectangular_2_phase(synced_q, synced_i) * AMSYNC_ERR_SCALE);
  //*err = (int64_t)(atan2(synced_q, synced_i) * AMSYNC_ONE);

  int64_t y0 = *err * AMSYNC_B0 + x1 * AMSYNC_B1 + x2 * AMSYNC_B2;
  y0 += y0_err;
  y0_err = y0 & AMSYNC_ONE;
  y0 >>= AMSYNC_FRACTION_BITS;
#if AMSYNC_NUM_TAPS == 3
  y0 += 2 * y1 - y2;
#else
  y0 += y1;
#endif
  y2 = y1;
  y1 = y0;
  x2 = x1;
  x1 = *err;
  phi_locked += y0;

  if(phi_locked > (2 * AMSYNC_PI))
  {
    phi_locked -= 2 * AMSYNC_PI;
  }

  if(phi_locked < (-2 * AMSYNC_PI))
  {
    phi_locked += 2 * AMSYNC_PI;
  }

  *i = vco_i;
  *q = vco_q;
}
"""


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


def pll_2nd_order_des(loop_bw):
    # prewarping
    wa = (2 * SR) * math.tan(2 * math.pi * loop_bw / (2 * SR))
    wn = wa / SR
    wn = 2 * math.pi * (loop_bw / SR)
    zeta = 1 / math.sqrt(2)
    K = 1
    t1 = K / (wn * wn)
    t2 = 2 * zeta / wn

    bf = [(2 * t2 + 1) / (2 * t1), (1 - 2 * t2) / (2 * t1)]
    af = [1, -1]

    return bf, af


def pll_3rd_order_des(loop_bw):
    # prewarping
    wa = (2 * SR) * math.tan(2 * math.pi * loop_bw / (2 * SR))
    wn = wa / SR
    wn = 2 * math.pi * (loop_bw / SR)
    b = 1.1
    c = 2.4

    bf = (
        b * wn**2 / 2 + c * wn + wn**3 / 4,
        -2 * c * wn + wn**3 / 2,
        -b * wn**2 / 2 + c * wn + wn**3 / 4,
    )
    af = [1, -2, 1]

    return bf, af


class PLL:

    def __init__(self, loop_bw):
        self.bf, self.af = pll_3rd_order_des(loop_bw)
        print(self.bf, self.af)
        self.num_taps = len(self.bf)

        if len(self.bf) < 3:
            self.bf += [0] * (3 - len(self.bf))

        if len(self.af) < 3:
            self.af += [0] * (3 - len(self.af))

        self.phi_locked = 0.0
        self.y1 = 0
        self.x1 = 0
        self.y2 = 0
        self.x2 = 0

    def process(self, i, q):
        out = complex(math.cos(self.phi_locked), math.sin(self.phi_locked))
        err = cmath.phase(complex(i, q) * complex(out.real, -out.imag))

        y0 = (
            err * self.bf[0]
            + self.x1 * self.bf[1]
            + self.x2 * self.bf[2]
            - self.y1 * self.af[1]
            - self.y2 * self.af[2]
        )
        self.y2 = self.y1
        self.y1 = y0
        self.x2 = self.x1
        self.x1 = err
        self.phi_locked += y0

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
    def __init__(self, loop_bw):

        self.bf, self.af = pll_3rd_order_des(loop_bw)
        print(self.bf, self.af)
        self.num_taps = len(self.bf)

        if len(self.bf) < 3:
            self.bf += [0] * (3 - len(self.bf))

        if len(self.af) < 3:
            self.af += [0] * (3 - len(self.af))

        self.bf = list(map(lambda x: round(x * FIX_ONE), self.bf))
        self.af = list(map(lambda x: round(x * FIX_ONE), self.af))

        self.phi_locked = 0
        self.y1 = 0
        self.x1 = 0
        self.y2 = 0
        self.x2 = 0
        self.y0_err = 0

        self.sin_table = []
        for idx in range(2048):
            self.sin_table.append(
                round(math.sin(2.0 * math.pi * idx / 2048.0) * BASE_MAX)
            )

    def process(self, i, q):

        idx = round(self.phi_locked / FIX_PHI_SCALE)
        if idx < 0:
            idx = 2048 + idx

        out_i = self.sin_table[((idx) + 512) & 0x7FF]
        out_q = self.sin_table[(idx) & 0x7FF]

        tmp_i = (i * out_i + q * out_q) >> BASE_BITS
        tmp_q = (-i * out_q + q * out_i) >> BASE_BITS

        err = np.int64((rectangular_2_phase(tmp_q, tmp_i) * FIX_ERR_SCALE))
        # err = np.int64(round(FIX_ONE * np.angle(tmp_i + 1j * tmp_q)))

        y0 = err * self.bf[0] + self.x1 * self.bf[1] + self.x2 * self.bf[2]
        y0 += self.y0_err
        self.y0_err = y0 & FIX_ONE
        y0 >>= FIX_ONE_BITS
        if self.num_taps == 3:
            y0 += 2 * self.y1 - self.y2
        else:
            y0 += self.y1
        self.y2 = self.y1
        self.y1 = y0
        self.x2 = self.x1
        self.x1 = err
        self.phi_locked += y0

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

        return out_i, out_q, err


def floating_sim(loop_bw, time, input):
    pll = PLL(loop_bw)

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


def fixed_sim(loop_bw, time, input):
    pll = PLLFixed(loop_bw)

    print(f"// PLL loop bandwidth: {loop_bw}Hz")
    print(f"#define AMSYNC_NUM_TAPS ({pll.num_taps})")
    for i in range(pll.num_taps):
        print(f"#define AMSYNC_B{i} ({pll.bf[i]})")
    for i in range(pll.num_taps):
        print(f"#define AMSYNC_A{i} ({pll.af[i]})")
    print(f"#define AMSYNC_PI ({FIX_PI})")
    print(f"#define AMSYNC_ONE ({FIX_ONE})")
    print(f"#define AMSYNC_MAX ({FIX_MAX})")
    print(f"#define AMSYNC_ERR_SCALE ({FIX_ERR_SCALE})")
    print(f"#define AMSYNC_PHI_SCALE ({FIX_PHI_SCALE})")
    print(f"#define AMSYNC_FRACTION_BITS ({FIX_ONE_BITS})")
    print(f"#define AMSYNC_BASE_FRACTION_BITS ({BASE_BITS})")

    output = []
    error = []
    for s in input:
        out_i, out_q, err = pll.process(
            round(np.real(s) * BASE_MAX), round(np.imag(s) * BASE_MAX)
        )
        output.append(complex(out_i, out_q))
        error.append(err)

    output = np.array(output)
    input1 = np.round(np.real(input) * BASE_MAX)

    plt.plot(time, input1, label="input real")
    plt.plot(time, np.real(output), label="output real")
    plt.plot(time, error, label="phase error")

    plt.grid(True)
    plt.legend()
    plt.show()


def c_fixed_sim(time, input):
    ffi = FFI()
    ffi.set_source(
        "_pll",
        SRC,
        source_extension=".c",
    )

    ffi.cdef(f"void amsync_demod(int16_t *i, int16_t *q, int64_t *err);")
    ffi.cdef("void initialise_luts(void);")
    ffi.compile()
    from _pll import lib as pll

    i = ffi.new("int16_t*")
    q = ffi.new("int16_t*")
    err = ffi.new("int64_t*")
    pll.initialise_luts()

    output = []
    error = []
    for s in input:
        i[0] = round(np.real(s) * BASE_MAX)
        q[0] = round(np.imag(s) * BASE_MAX)
        pll.amsync_demod(i, q, err)
        output.append(complex(i[0], q[0]))
        error.append(err[0])

    output = np.array(output)
    input1 = np.round(np.real(input) * BASE_MAX)

    plt.plot(time, input1, label="input real")
    plt.plot(time, np.real(output), label="output real")
    plt.plot(time, error, label="phase error")

    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    end = 0.8
    time = np.arange(0, end, 1 / SR)
    # freq = np.linspace(10, 100, len(time))
    freq = 50 + 10 * np.sin(2 * np.pi * 3 * time)
    phase = 2 * np.pi * freq * time + 10
    phase += (time > (end / 4)) * 1
    phase -= (time > (end / 2)) * 1
    phase += (time > (3 * end / 4)) * 1
    input_a = 0.3 * np.exp(1j * phase)
    input_a += (
        np.random.random(len(time)) * 0.01 + 1j * np.random.random(len(time)) * 0.01
    )

    floating_sim(50, time, input_a)
    fixed_sim(50, time, input_a)
    c_fixed_sim(time, input_a)
