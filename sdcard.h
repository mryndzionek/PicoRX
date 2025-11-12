#pragma once

#include <cstdint>

bool sdcard_init(uint32_t c);

uint32_t sdcard_start_recording(uint32_t frequency, uint8_t mode);
void sdcard_stop_recording(void);

void sdcard_write(uint16_t const* const data, uint16_t n);
bool sdcard_needs_flush(void);
void sdcard_flush(void);
