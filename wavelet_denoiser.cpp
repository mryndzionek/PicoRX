// Copyright (c) Mariusz Ryndzionek 2025
// https://github.com/mryndzionek
// License: MIT
//

#include "wavelet_denoiser.h"

#include <assert.h>
#include <math.h>
#include <pico.h>

#include <climits>
#include <cstdio>

typedef int16_t s_num_t;
typedef int32_t d_num_t;

#define FRAME_LEN (64 + 16)
#define OVERLAP_LEN (16)

#define LEVELS (3)
#define CD_LEN_ALL (101)

#define MAX_THRESHOLD_LEVEL (6)

#define extra_bits (6)

#define HN (16)

#define ULONG_BITS (sizeof(unsigned long) * CHAR_BIT)

#define FINEST_LEVEL_DETAIL_LEN (23)  // last element of CD_LEN_LUT
#define MEDIAN_AVG_LEN (FINEST_LEVEL_DETAIL_LEN * 120)

#if 1
// sym8 wavelet filters
static const s_num_t H_LO[HN] = {62,    -10,   -490,  125,   1610,  -892,
                                 -1702, 11942, 25466, 15773, -2008, -4695,
                                 249,   1039,  -18,   -111};
static const s_num_t H_HI[HN] = {-111,  18,     1039,  -249, -4695, 2008,
                                 15773, -25466, 11942, 1702, -892,  -1610,
                                 125,   490,    -10,   -62};
static const s_num_t H_LO_INV[HN] = {-111,  -18,   1039,  249,   -4695, -2008,
                                     15773, 25466, 11942, -1702, -892,  1610,
                                     125,   -490,  -10,   62};
static const s_num_t H_HI_INV[HN] = {-62,  -10,   490,    125,   -1610, -892,
                                     1702, 11942, -25466, 15773, 2008,  -4695,
                                     -249, 1039,  18,     -111};
#else
// db8 wavelet filters
static const s_num_t H_LO[HN] = {1783, 10252, 22138, 19180, -519, -9306,
                                 15,   4219,  -569,  -1445, 458,  287,
                                 -160, -13,   22,    -4};
static const s_num_t H_HI[HN] = {-4,    -22,    -13,   160,  287,   -458,
                                 -1445, 569,    4219,  -15,  -9306, 519,
                                 19180, -22138, 10252, -1783};
static const s_num_t H_LO_INV[HN] = {-4,    22,    -13,   -160, 287,   458,
                                     -1445, -569,  4219,  15,   -9306, -519,
                                     19180, 22138, 10252, 1783};
static const s_num_t H_HI_INV[HN] = {-1783, 10252, -22138, 19180, 519,  -9306,
                                     -15,   4219,  569,    -1445, -458, 287,
                                     160,   -13,   -22,    -4};
#endif

// one-sided cosine window
static const int16_t WIN[OVERLAP_LEN] = {
    0,     358,   1416,  3129,  5421,  8192,  11321, 14671,
    18096, 21446, 24575, 27346, 29638, 31351, 32409, 32767};

static const s_num_t CD_LEN_LUT[LEVELS] = {47, 31, 23};
static const s_num_t CD_INV_LEN_LUT[LEVELS] = {31, 47, 79};

static uint8_t g_threshold = 0;

static inline d_num_t product(s_num_t a, s_num_t b) {
  return ((d_num_t)a * b) >> (15 - extra_bits);
}

static inline d_num_t __time_critical_func(mac)(s_num_t const *const x,
                                                s_num_t const *const h,
                                                uint16_t n) {
  d_num_t accum = 0;

  for (uint16_t i = 0; i < 2 * n; i += 2) {
    accum += product(x[i], h[i]);
  }

  return accum / (1 << extra_bits);
}

static inline d_num_t __time_critical_func(mac_inv)(s_num_t const *const x,
                                                    s_num_t const *const h,
                                                    uint16_t n) {
  d_num_t accum = 0;

  for (uint16_t i = 0; i < n; i++) {
    accum += product(x[i], h[2 * i]);
  }

  return accum / (1 << extra_bits);
}

static void __time_critical_func(convolve)(s_num_t const *const x, uint16_t xn,
                                           s_num_t const *const h, uint16_t hn,
                                           s_num_t *y) {
  for (uint16_t i = 0; i < 2 * xn; i += 2) {
    if (i + 2 * hn > 2 * xn) {
      y[i] = mac(&x[i], h, xn - (i / 2));
    } else {
      y[i] = mac(&x[i], h, hn);
    }
  }
}

static void __time_critical_func(convolve_inv)(s_num_t const *const x,
                                               uint16_t xn,
                                               s_num_t const *const h,
                                               uint16_t hn, s_num_t *y) {
  for (uint16_t i = 0; i < xn; i++) {
    if (i + hn > xn) {
      y[2 * i] = mac_inv(&x[i], h, xn - i);
    } else {
      y[2 * i] = mac_inv(&x[i], h, hn);
    }
  }
}

static void dwt(s_num_t const *const x, uint16_t xn, s_num_t const *const h_lo,
                s_num_t const *const h_hi, uint16_t hn, s_num_t *ca,
                s_num_t *cd) {
  convolve(&x[0], xn / 2, &h_lo[0], hn / 2, &ca[0]);
  convolve(&x[1], xn / 2, &h_lo[1], hn / 2, &ca[1]);
  convolve(&x[0], xn / 2, &h_hi[0], hn / 2, &cd[0]);
  convolve(&x[1], xn / 2, &h_hi[1], hn / 2, &cd[1]);
  for (uint16_t i = 0; i < (xn / 2) - 1; i++) {
    ca[i] = ca[2 * (i + 1)] + ca[2 * (i + 1) + 1];
    cd[i] = cd[2 * (i + 1)] + cd[2 * (i + 1) + 1];
  }
}

static void idwt(s_num_t const *const ca, s_num_t const *const cd, uint16_t cn,
                 s_num_t const *const h_lo, s_num_t const *const h_hi,
                 uint16_t hn, s_num_t *x) {
  s_num_t tmp_x[2 * cn];

  convolve_inv(ca, cn, &h_lo[0], hn / 2, &x[1]);
  convolve_inv(ca, cn, &h_lo[1], hn / 2, &x[0]);
  convolve_inv(cd, cn, &h_hi[0], hn / 2, &tmp_x[1]);
  convolve_inv(cd, cn, &h_hi[1], hn / 2, &tmp_x[0]);

  for (uint16_t i = 0; i < 2 * cn; i++) {
    x[i] += tmp_x[i];
  }
}

static void dwtn(s_num_t *const x, uint16_t xn, s_num_t const *const h_lo,
                 s_num_t const *const h_hi, uint16_t hn, s_num_t *ca,
                 s_num_t *cd) {
  uint16_t idx = 0;
  s_num_t _ca[xn];
  s_num_t _cd[xn];

  for (uint16_t i = 0; i < LEVELS; i++) {
    dwt(x, xn, h_lo, h_hi, hn, _ca, _cd);
    xn = CD_LEN_LUT[i];
    // use approx. as new input
    idx += xn;
    for (uint16_t j = 0; j < xn; j++) {
      x[hn + j] = _ca[j];                 // first [hn] is pad
      cd[CD_LEN_ALL - idx + j] = _cd[j];  // store details
    }
    xn += hn;
    if (xn % 2 == 1) {
      x[xn++] = 0;
    }
  }

  for (uint16_t i = 0; i < CD_LEN_LUT[LEVELS - 1]; i++) {
    ca[i] = _ca[i];
  }
}

static void idwtn(s_num_t *const ca, s_num_t *const cd, uint16_t cn,
                  s_num_t const *const h_lo, s_num_t const *const h_hi,
                  uint16_t hn, s_num_t *x) {
  s_num_t const *cd_p = cd;

  for (uint16_t i = 0; i < LEVELS; i++) {
    idwt(ca, cd_p, cn, h_lo, h_hi, hn, x);
    cd_p += cn;
    cn = CD_INV_LEN_LUT[i];
    for (uint16_t j = 0; j < cn; j++) {
      ca[j] = x[j];
    }
  }
}

static inline void __time_critical_func(soft_threshold)(s_num_t *data,
                                                        d_num_t thr,
                                                        uint16_t n) {
  for (uint16_t i = 0; i < n; i++) {
    if (abs(data[i]) < thr) {
      data[i] = 0;
    } else if (data[i] > 0) {
      data[i] -= thr;
    } else {
      data[i] += thr;
    }
  }
}

static inline int _cmp(const void *a, const void *b) {
  if (*(uint16_t *)a > *(uint16_t *)b) {
    return -1;
  }
  if (*(uint16_t *)a < *(uint16_t *)b) {
    return 1;
  }
  return 0;
}

static s_num_t estimate_noise_level(s_num_t x[FINEST_LEVEL_DETAIL_LEN]) {
  static uint16_t medians[MEDIAN_AVG_LEN];
  static uint16_t med_idx = 0;
  static s_num_t sigma = 0;

  for (uint16_t i = 0; i < FINEST_LEVEL_DETAIL_LEN; i++) {
    medians[med_idx++] = abs(x[i]);
  }

  if (med_idx == MEDIAN_AVG_LEN) {
    med_idx = 0;
    qsort(medians, MEDIAN_AVG_LEN, sizeof(medians[0]), _cmp);
    const d_num_t median =
        (medians[MEDIAN_AVG_LEN / 2] + medians[(MEDIAN_AVG_LEN / 2) - 1]) / 2;
    sigma = median + (((d_num_t)median * 15813) >> 15);  // divide by 0.674
  }
  return sigma;
}

void wavelet_set_threshold(uint8_t thres) {
  if (thres > MAX_THRESHOLD_LEVEL) {
    thres = MAX_THRESHOLD_LEVEL;
  } else if (thres == 0) {
    thres = 1;
  }

  g_threshold = thres;
}

bool wavelet_denoise(int16_t audio[64], bool output_denoised) {
  static s_num_t prev_in[OVERLAP_LEN];
  static s_num_t prev_out[OVERLAP_LEN];
  static d_num_t y1 = 0;

  s_num_t x[FRAME_LEN + HN] = {0};
  s_num_t ca[FRAME_LEN + HN] = {0};
  s_num_t cd[CD_LEN_ALL] = {0};

  for (uint16_t i = 0; i < OVERLAP_LEN; i++) {
    x[HN + i] = prev_in[i];
  }

  for (uint16_t i = 0; i < 64; i++) {
    x[HN + i + OVERLAP_LEN] = audio[i];
  }

  for (uint16_t i = 0; i < OVERLAP_LEN; i++) {
    prev_in[i] = audio[64 - OVERLAP_LEN + i];
  }

  dwtn(x, FRAME_LEN + HN, H_LO, H_HI, HN, ca, cd);

  d_num_t ca_sum = 0;
  d_num_t cd_sum = 0;
  for (uint16_t i = 0; i < FINEST_LEVEL_DETAIL_LEN; i++) {
    ca_sum += ca[i] * ca[i];
    cd_sum += cd[i] * cd[i];
  }

  if (output_denoised) {
    const s_num_t sigma = estimate_noise_level(cd);
    const d_num_t threshold = (2 * (g_threshold + 1)) * sigma;  // VisuShrink
    soft_threshold(cd, threshold, CD_LEN_ALL);  // soft threshold
    idwtn(ca, cd, CD_LEN_LUT[LEVELS - 1], H_LO_INV, H_HI_INV, HN, x);

    for (uint16_t i = 0; i < OVERLAP_LEN; i++) {
      // blend overlaps
      prev_out[i] = ((d_num_t)prev_out[i] * (32767 - WIN[i])) >> 15;
      x[i] = prev_out[i] + (((d_num_t)x[i] * WIN[i]) >> 15);
      prev_out[i] = x[FRAME_LEN - OVERLAP_LEN + i];
    }

    for (uint16_t i = 0; i < 64; i++) {
      audio[i] = x[i];
    }
  }

  d_num_t x0 = ca_sum > cd_sum ? ((ca_sum - cd_sum) >> 4) : 0;
  if (x0 > 32767) {
    x0 = 32767;
  }
  const d_num_t y0 = y1 + ((x0 - y1) >> 4);
  y1 = y0;

  printf("%ld\n", y0);

  return ((y0 > 2500) ||
          (x0 > 5000));  // TODO adjust this threshold, or make it configurable?
}
