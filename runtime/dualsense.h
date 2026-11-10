#ifndef DUALSENSE_H
#define DUALSENSE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS_SFX_NONE = 0,
    DS_SFX_BOW_PULL,
    DS_SFX_ARROW_SHOOT,
    DS_SFX_SHIELD_BLOCK,
    DS_SFX_WEAPON_HIT,
    DS_SFX_FAST_TRAVEL,
    DS_SFX_CHEST_OPEN,
    DS_SFX_COUNT
} DualSenseSfx;

// Initialize the DualSense controller speaker audio, haptics, and trigger pipeline
void dualsense_init(void);

// Poll controller input state for adaptive triggers (bow tension, shield resistance) and shot detection
void dualsense_poll(void *sdl_gamepad, const uint8_t *pd, int16_t l2_raw, int16_t r2_raw);

// Called when the game engine sends HD Rumble vibration values
void dualsense_on_vibration(int side, float amp_low, float freq_low, float amp_high, float freq_high);

// Feed live in-game rendered audio frames directly to DualSense speaker and voice-coil haptics
void dualsense_feed_audio(const float *lr, int frames, int rate);

// Trigger a specific interactive sound effect and haptic pulse on the DualSense
void dualsense_play_sfx(DualSenseSfx sfx);

// Check if a DualSense controller with speaker audio & adaptive triggers is actively connected
bool dualsense_active(void);

// SDL's HID backend obtains the DualSense battery state from its active controller report. Called under input.c's lock.
int dualsense_power_info(void *sdl_gamepad, int *percent, int *charging);
void dualsense_power_reset(void);

// Shutdown and release audio streams
void dualsense_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // DUALSENSE_H
