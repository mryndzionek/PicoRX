#include "rnn_denoiser.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fastgrnn_fc_params.h"
#include "fastgrnn_rnn_params.h"
#include "pico/stdlib.h"

#define MEL_BINS (16)
#define FB_LEN (6)

// clang-format off

const uint8_t FILTERBANK_OFF[MEL_BINS] =
   { 1,  2,  3,  4,  5,  6,  7,  8,
     9,  10,  11,  13,  14,  16,  18,  20};

const float FILTERBANK_MAT[MEL_BINS][FB_LEN] = {
   { 9.31625545e-01,  6.83744699e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 8.97438288e-01,  1.02561705e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 8.63251090e-01,  1.36748940e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 8.29063833e-01,  1.70936182e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 7.94876575e-01,  2.05123410e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 7.60689378e-01,  2.39310652e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 7.26502120e-01,  2.73497880e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 6.66147709e-01,  3.33852321e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 5.52752554e-01,  4.47247416e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 3.74471605e-01,  5.86285770e-01,  3.92426066e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 1.51941389e-01,  5.86315572e-01,  2.61743009e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 3.42631578e-01,  4.91506606e-01,  1.65861830e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00},
   { 8.95816013e-02,  3.72176290e-01,  3.93779665e-01,  1.44462436e-01,  0.00000000e+00,  0.00000000e+00},
   { 1.09104842e-01,  3.31465364e-01,  3.77802938e-01,  1.81626856e-01,  0.00000000e+00,  0.00000000e+00},
   { 6.96612746e-02,  2.35297531e-01,  3.77811849e-01,  2.31680393e-01,  8.55489373e-02,  0.00000000e+00},
   { 9.94521007e-03,  1.44042730e-01,  2.78140247e-01,  3.07597220e-01,  1.89290598e-01,  7.09839761e-02},
};

const float MEAN_VEC[MEL_BINS] =
   {-2.62648058e+00, -2.55596542e+00, -2.55052757e+00, -2.55786133e+00, -2.58404779e+00, -2.65054488e+00, -2.71725464e+00, -2.76605701e+00,
    -2.80118346e+00, -2.82043195e+00, -2.82704830e+00, -2.84161854e+00, -2.86585426e+00, -2.90588737e+00, -2.93292713e+00, -2.91756248e+00};

const float STD_VEC[MEL_BINS] =
   { 6.94419384e-01,  7.60908425e-01,  7.57894278e-01,  7.58915484e-01,  7.55334556e-01,  7.37524152e-01,  7.30422378e-01,  7.19806731e-01,
     7.02844501e-01,  6.83217466e-01,  6.67753160e-01,  6.50811791e-01,  6.18354201e-01,  6.11369669e-01,  6.04896069e-01,  5.97592235e-01};

const uint8_t MEL_IDX[MEL_BINS] =
   { 0,  1,  2,  4,  5,  6,  7,  8,
     10,  11,  13,  15,  17,  20,  23,  26};

// clang-format on

static inline rnn_num_t sigmoidf(rnn_num_t x) { return (1 / (1 + expf(-x))); }

static void __time_critical_func(_rnn_process)(const rnn_num_t input[16],
                                               const rnn_num_t hidden[16],
                                               rnn_num_t output[16]) {
  rnn_num_t z;
  rnn_num_t c;

  for (uint16_t j = 0; j < 8; j++) {
    for (uint16_t i = 0; i < 16; i++) {
      output[j] += (GRNN0_W[j][i] * input[i]);
    }
  }

  for (uint16_t j = 0; j < 8; j++) {
    for (uint16_t i = 0; i < 8; i++) {
      output[j] += (GRNN0_U[j][i] * hidden[i]);
    }
  }

  for (uint16_t j = 0; j < 8; j++) {
    z = output[j] + GRNN0_BIAS_GATE[j];
    z = sigmoidf(z);
    c = output[j] + GRNN0_BIAS_UPDATE[j];
    c = tanhf(c);

    output[j] =
        z * hidden[j] + (GRNN0_SIGM_ZETA * (1.0 - z) + GRNN0_SIGM_NU) * c;
  }
}

static void __time_critical_func(fc_process)(const rnn_num_t input[8],
                                             rnn_num_t output[16]) {
  memset(output, 0, 16 * sizeof(rnn_num_t));

  for (size_t j = 0; j < 16; j++) {
    for (size_t i = 0; i < 8; i++) {
      output[j] += (input[i] * FC_W[j][i]);
    }
    output[j] += FC_B[j];
  }
}

static void rnn_process(const rnn_num_t input[16], rnn_num_t output[8]) {
  static rnn_num_t hidden[8] = {0};

  memset(output, 0, sizeof(rnn_num_t) * 8);
  _rnn_process(input, hidden, output);
  memcpy(hidden, output, sizeof(rnn_num_t) * 8);
}

void rnn_denoiser_denoise(uint16_t x[RNND_NFFT], rnn_num_t g[RNND_NFFT]) {
  rnn_num_t out[16] = {0};
  rnn_num_t input[MEL_BINS];

  for (uint16_t i = 0; i < MEL_BINS; i++) {
    rnn_num_t s = 0;
    for (uint16_t j = 0; j < FB_LEN; j++) {
      if (FILTERBANK_MAT[i][j] > 0) {
        s += FILTERBANK_MAT[i][j] * x[FILTERBANK_OFF[i] + j];
      }
    }
    input[i] = ((log10f(1e-8f + (s / 3276.7)) - MEAN_VEC[i]) / STD_VEC[i]);
  }

  rnn_process(input, out);
  fc_process(out, input);

  memset(g, 0, RNND_NFFT * sizeof(rnn_num_t));

  for (uint16_t i = 0; i < MEL_BINS; i++) {
    input[i] = (input[i] + 1) / 2;
  }

  // interpolate
  int16_t x1 = MEL_IDX[0];
  rnn_num_t y1 = input[0];
  uint16_t k = 0;

  for (uint16_t i = 1; i < MEL_BINS + 1; i++) {
    const int16_t x0 = x1;
    const rnn_num_t y0 = y1;
    if (i < MEL_BINS) {
      y1 = input[i];
      x1 = MEL_IDX[i];
    } else {
      x1++;
    }
    const rnn_num_t d = (y1 - y0) / (x1 - x0);
    for (uint16_t j = 0; j < (x1 - x0); j++) {
      g[k] = y0 + j * d;
      if (g[k] > 1.0f) {
        g[k] = 1.0f;
      }
      if (g[k] < 0.0f) {
        g[k] = 0.0f;
      }
      k++;
    }
  }

  return;
}
