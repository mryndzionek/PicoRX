// Copyright (c) Mariusz Ryndzionek 2025
// https://github.com/mryndzionek
// License: MIT
//

#pragma once

#include <stdint.h>

void wavelet_denoise(int16_t audio[64]);
void wavelet_set_threshold(uint8_t tres);
