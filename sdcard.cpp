#include "sdcard.h"

#include "f_util.h"
#include "ff.h"
#include "hw_config.h"
#include "ring_buffer_lib.h"
#include "utils.h"

#define SD_BLOCK_SIZE \
  (1024)  // must be multiple of 512 for maximum FAT32 write speed

// from time to time (~5 minutes with a 16GB card) the write
// speed slows down by a factor of 100, or more
// and only with this buffer size we can accommodate
#define BUF_SIZE (30 * SD_BLOCK_SIZE)

// #define SD_DBG

#ifdef SD_DBG
#define SD_DBG_PRINTF(fmt, args...) \
  printf("DEBUG: %d:%s(): " fmt "\n", __LINE__, __func__, ##args)
#else
#define SD_DBG_PRINTF(fmt, args...) \
  do {                              \
  } while (0)
#endif

static const uint8_t wav_header[44] = {
    0x52, 0x49, 0x46, 0x46, 0x18, 0x20, 0xe9, 0x02, 0x57, 0x41, 0x56,
    0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x01, 0x00, 0x98, 0x3a, 0x00, 0x00, 0x30, 0x75, 0x00, 0x00, 0x02,
    0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00};

static sd_sdio_if_t sdio_if = {
    .CMD_gpio = 12,
    .D0_gpio = 7,
    .SDIO_PIO = pio1,
    .DMA_IRQ_num = DMA_IRQ_1,
    .baud_rate =
        8000000,  // can be increased to ~20MHz with a properly designed PCB
};

static sd_card_t sd_card = {.type = SD_IF_SDIO, .sdio_if_p = &sdio_if};
size_t sd_get_num() { return 1; }
sd_card_t* sd_get_by_num(size_t num) {
  if (0 == num) {
    // The number 0 is a valid SD card number.
    // Return a pointer to the sd_card object.
    return &sd_card;
  } else {
    // The number is invalid. Return @c NULL.
    return NULL;
  }
}

static bool card_mounted = false;
static bool writing_enabled = false;
static FIL file;
static uint32_t f_count;
static uint32_t b_count;
static ring_buffer_t sdcard_rb;
static uint8_t usb_buf[BUF_SIZE];
static FATFS fs;

static void update_wav_size(void) {
  unsigned int bw;
  FRESULT fr = f_lseek(&file, 40);
  if (fr != FR_OK) {
    SD_DBG_PRINTF("seek failed: %d", fr);
    return;
  }
  fr = f_write(&file, &b_count, sizeof(b_count), &bw);
  if (fr != FR_OK) {
    SD_DBG_PRINTF("write failed: %d", fr);
    return;
  }
  fr = f_lseek(&file, f_size(&file));
  if (fr != FR_OK) {
    SD_DBG_PRINTF("seek failed: %d", fr);
    return;
  }
}

bool sdcard_init(uint32_t c) {
  f_count = c + 1;
  FRESULT fr = f_mount(&fs, "", 1);
  if (FR_OK == fr) {
    int sl = spin_lock_claim_unused(true);
    ring_buffer_init(&sdcard_rb, usb_buf, BUF_SIZE, sl);
    card_mounted = true;
  } else {
    SD_DBG_PRINTF("mount failed: %d", fr);
  }
  return card_mounted;
}

uint32_t sdcard_start_recording(uint32_t frequency, uint8_t mode) {
  if (!card_mounted) {
    return f_count;
  }

  if (writing_enabled) {
    f_close(&file);
  }

  while (true) {
    char filename[32];
    snprintf(filename, sizeof(filename), "rec_%06ld_%ld_%s.wav", f_count,
             frequency, mode_to_str(mode));
    FRESULT fr = f_open(&file, filename, FA_CREATE_NEW | FA_WRITE);
    if (FR_EXIST == fr) {
      f_count++;
      if (f_count == 999999) {
        f_count = 0;
      }
    } else if (FR_OK == fr) {
      SD_DBG_PRINTF("Opening file: %s", filename);
      unsigned int bw;
      fr = f_write(&file, wav_header, 44, &bw);
      writing_enabled = true;
      b_count = 0;
      f_count++;
      if (f_count == 999999) {
        f_count = 0;
      }
      break;
    } else {
      break;
    }
  }

  return f_count;
}

void sdcard_stop_recording(void) {
  if (!card_mounted) {
    return;
  }

  update_wav_size();
  FRESULT fr = f_close(&file);
  if (fr != FR_OK) {
    SD_DBG_PRINTF("close failed: %d", fr);
  }
}

void sdcard_write(uint16_t const* const data, uint16_t n) {
  ring_buffer_push_ovr(&sdcard_rb, (uint8_t*)data, 2 * n);
}

bool sdcard_needs_flush(void) {
  if (!card_mounted) {
    return false;
  } else {
    return ring_buffer_get_num_bytes(&sdcard_rb) >= SD_BLOCK_SIZE;
  }
}

void sdcard_flush(void) {
  static uint16_t fc = 0;
  unsigned int bw;
  FRESULT fr;
  static uint8_t buf[SD_BLOCK_SIZE];

  if (!card_mounted) {
    return;
  }

  if (writing_enabled) {
    int32_t b = ring_buffer_get_num_bytes(&sdcard_rb);
    const uint32_t start = time_us_32();
    while ((b >= SD_BLOCK_SIZE) && ((time_us_32() - start) < 1800000)) {
      uint32_t tm = time_us_32();
      ring_buffer_pop(&sdcard_rb, buf, SD_BLOCK_SIZE);
      fr = f_write(&file, buf, SD_BLOCK_SIZE, &bw);
      const uint32_t dur = time_us_32() - tm;
      if (dur > 20000) {
        SD_DBG_PRINTF("w: %d %ld %d %ld %ld", fr, b, bw, dur,
                      time_us_32() - start);
      }
      if (FR_OK == fr) {
        b_count += SD_BLOCK_SIZE;
        b -= SD_BLOCK_SIZE;
        fc++;
      } else {
        SD_DBG_PRINTF("write error: %d", fr);
      }
    }
    if (fc >= 1024) {  // every ~1MB update and sync file
      fc = 0;
      update_wav_size();
      fr = f_sync(&file);
      if (fr != FR_OK) {
        SD_DBG_PRINTF("sync error: %d", fr);
      }
    }
  }
}
