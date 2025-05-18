#include "utils.h"
#include <cstdint>
#include <math.h>

#define CORDIC_ITERS (6)

int16_t sin_table[2048];
static const uint32_t CORDIC_GAIN = 39803;
static const int16_t CORDIC_ATAN_LUT[CORDIC_ITERS] = {8192, 4836, 2555, 1297, 651, 326};

void rectangular_2_polar(int16_t i, int16_t q, uint16_t *mag, int16_t *phase) {
  int16_t temp_i;
  int16_t angle = 0;

  if (i < 0) {
    // rotate by an initial +/- 90 degrees
    temp_i = i;
    if (q > 0.0f) {
      i = q; // subtract 90 degrees
      q = -temp_i;
      angle = 16384;
    } else {
      i = -q; // add 90 degrees
      q = temp_i;
      angle = -16384;
    }
  }

  for (uint16_t k = 0; k < CORDIC_ITERS; k++) {
    temp_i = i;
    if (q > 0) {
      /* Rotate clockwise */
      i += (q >> k);
      q -= (temp_i >> k);
      angle += CORDIC_ATAN_LUT[k];
    } else {
      /* Rotate counterclockwise */
      i -= (q >> k);
      q += (temp_i >> k);
      angle -= CORDIC_ATAN_LUT[k];
    }
  }

  *mag = ((uint32_t)i * CORDIC_GAIN) >> 16;
  *phase = angle;
}

void initialise_luts()
{
  //pre-generate sin/cos lookup tables
  float scaling_factor = (1 << 15) - 1;
  for(uint16_t idx=0; idx<2048; idx++)
  {
    sin_table[idx] = roundf(sinf(2.0*M_PI*idx/2048.0) * scaling_factor);
  }
}
