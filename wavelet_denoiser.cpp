// Copyright (c) Mariusz Ryndzionek 2025
// https://github.com/mryndzionek
// License: MIT
//

#include "wavelet_denoiser.h"

#include <assert.h>
#include <math.h>
#include <pico.h>

typedef int16_t s_num_t;
typedef int32_t d_num_t;

#define FRAME_LEN (64)
#define LEVELS (3)
#define CD_LEN_ALL (87)

#define extra_bits (3)

// sym8 wavelet filters
#define HN (16)
#if 1
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

static const s_num_t CD_LEN_LUT[LEVELS] = {39, 27, 21};
static const s_num_t CD_INV_LEN_LUT[LEVELS] = {27, 39, 63};

static uint16_t g_threshold = 9500;

static inline d_num_t product(s_num_t a, s_num_t b) {
  return (a * b) >> (15 - extra_bits);
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

static inline int _cmp(const void *a, const void *b) {
  if (*(s_num_t *)a > *(s_num_t *)b) {
    return -1;
  }
  if (*(s_num_t *)a < *(s_num_t *)b) {
    return 1;
  }
  return 0;
}

static inline void __time_critical_func(soft_threshold)(s_num_t *data,
                                                        s_num_t thr,
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

void wavelet_set_threshold(uint8_t tres) {
  if (tres > 5) {
    tres = 5;
  }

  g_threshold = (6 - tres);
}

void wavelet_denoise(int16_t audio[FRAME_LEN]) {
  s_num_t x[FRAME_LEN + HN] = {0};
  s_num_t ca[FRAME_LEN + HN] = {0};
  s_num_t cd[CD_LEN_ALL] = {0};

  const uint16_t cd_level1_len = CD_LEN_LUT[0];
  s_num_t tmp[cd_level1_len];

  for (uint16_t i = 0; i < FRAME_LEN; i++) {
    x[HN + i] = audio[i];
  }

  dwtn(x, FRAME_LEN + HN, H_LO, H_HI, HN, ca, cd);

  for (uint16_t i = 0; i < cd_level1_len; i++) {
    tmp[i] = abs(cd[CD_LEN_ALL - cd_level1_len + i]);  // Level 1 is last
  }
  qsort(tmp, cd_level1_len, sizeof(tmp[0]), _cmp);
  d_num_t median = tmp[(cd_level1_len / 2)];

  d_num_t threshold = 64 * median / g_threshold;  // VisuShrink
  soft_threshold(cd, threshold, CD_LEN_ALL);      // soft threshold
  idwtn(ca, cd, CD_LEN_LUT[LEVELS - 1], H_LO_INV, H_HI_INV, HN, x);

  for (uint16_t i = 0; i < FRAME_LEN; i++) {
    audio[i] = x[i];
  }
}
