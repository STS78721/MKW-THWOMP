#include "dualsense.h"
#include "host.h"
#include "input_map.h"
#include "input.h"
#include "cfg.h"
#include "platform/platform.h"
#include <SDL3/SDL.h>
#include <stdatomic.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define PI 3.14159265358979323846f
#define SAMPLE_RATE 48000

// DualSense HID output report structure for adaptive triggers & haptics
typedef struct {
    uint8_t ucEnableBits1;              /* 0 */
    uint8_t ucEnableBits2;              /* 1 */
    uint8_t ucRumbleRight;              /* 2 */
    uint8_t ucRumbleLeft;               /* 3 */
    uint8_t ucHeadphoneVolume;          /* 4 */
    uint8_t ucSpeakerVolume;            /* 5 */
    uint8_t ucMicrophoneVolume;         /* 6 */
    uint8_t ucAudioEnableBits;          /* 7 */
    uint8_t ucMicLightMode;             /* 8 */
    uint8_t ucAudioMuteBits;            /* 9 */
    uint8_t rgucRightTriggerEffect[11]; /* 10 */
    uint8_t rgucLeftTriggerEffect[11];  /* 21 */
    uint8_t rgucUnknown1[6];            /* 32 */
    uint8_t ucLedFlags;                 /* 38 */
    uint8_t rgucUnknown2[2];            /* 39 */
    uint8_t ucLedAnim;                  /* 41 */
    uint8_t ucLedBrightness;            /* 42 */
    uint8_t ucPadLights;                /* 43 */
    uint8_t ucLedRed;                   /* 44 */
    uint8_t ucLedGreen;                 /* 45 */
    uint8_t ucLedBlue;                  /* 46 */
} DS5EffectsState_t;

static SDL_AudioStream *g_ds_stream = NULL;
static SDL_AudioDeviceID g_ds_device = 0;
static bool g_ds_active = false;
static PlatMutex g_ds_lock = PLAT_MUTEX_INIT;
static SDL_Gamepad *g_current_pad = NULL;
static SDL_Gamepad *g_battery_pad = NULL;
static SDL_hid_device *g_battery_hid = NULL;
static int g_battery_percent = -1, g_battery_charging;
static uint64_t g_battery_retry_at;

// State tracking for triggers & haptics
static bool g_bow_drawn = false;
static uint64_t g_bow_drawn_at = 0;
static uint64_t g_bow_recoil_until = 0;
static bool g_shield_raised = false;
static uint64_t g_shield_kickback_until = 0;
static int16_t g_last_r2 = 0;
static int16_t g_last_l2 = 0;
static uint64_t g_last_block_time = 0;

static uint8_t g_current_r2_mode = 0xFF;
static uint8_t g_current_l2_mode = 0xFF;

// Speaker activity gate & envelope (suppresses continuous BGM/ambience, unmutes for bow/shield/hits/SFX)
static _Atomic uint64_t g_speaker_active_until = 0;
static float g_gate_gain = 0.0f;

// Bandpass filter states for speaker (High-pass 280 Hz, Low-pass 6200 Hz)
static float g_hp_x1_l = 0.0f, g_hp_y1_l = 0.0f;
static float g_hp_x1_r = 0.0f, g_hp_y1_r = 0.0f;
static float g_spk_lp_y1_l = 0.0f, g_spk_lp_y1_r = 0.0f;

// Low-pass filter for voice coils (140 Hz)
static float g_lp_y1_l = 0.0f, g_lp_y1_r = 0.0f;
static int g_ds_channels = 4;

static bool dualsense_is_ps5_gamepad(SDL_Gamepad *pad) {
    if (!pad) return false;
    SDL_GamepadType type = SDL_GetGamepadType(pad);
    if (type == SDL_GAMEPAD_TYPE_PS5) return true;
    type = SDL_GetRealGamepadType(pad);
    if (type == SDL_GAMEPAD_TYPE_PS5) return true;
    const char *name = SDL_GetGamepadName(pad);
    if (name && (strstr(name, "DualSense") || strstr(name, "dualsense") || strstr(name, "PS5"))) return true;
    return false;
}

static void dualsense_battery_close(void) {
    if (g_battery_hid) SDL_hid_close(g_battery_hid);
    g_battery_hid = NULL;
    g_battery_pad = NULL;
    g_battery_percent = -1;
    g_battery_charging = 0;
    g_battery_retry_at = 0;
}

// Open a second, nonblocking HID read handle for the same controller SDL selected. This preserves SDL's gamepad and
// output handles (including the speaker, haptics and adaptive triggers) while exposing the status byte SDL omits.
static void dualsense_battery_open(SDL_Gamepad *pad) {
    dualsense_battery_close();
    if (!pad || !dualsense_is_ps5_gamepad(pad)) return;
    g_battery_pad = pad;
    g_battery_retry_at = plat_now_ns() + 5000000000ull;
    SDL_JoystickID id = SDL_GetGamepadID(pad);
    const char *active_path = SDL_GetGamepadPathForID(id);
    SDL_hid_device_info *devices = SDL_hid_enumerate(SDL_GetGamepadVendor(pad), SDL_GetGamepadProduct(pad));
    SDL_hid_device_info *match = NULL;
    for (SDL_hid_device_info *dev = devices; dev; dev = dev->next) {
        if (active_path && dev->path && SDL_strcasecmp(active_path, dev->path) == 0) { match = dev; break; }
        if (!active_path && !match) match = dev;
    }
    if (match && match->path) g_battery_hid = SDL_hid_open_path(match->path);
    SDL_hid_free_enumeration(devices);
    if (g_battery_hid && SDL_hid_set_nonblocking(g_battery_hid, 1) != 0) {
        SDL_hid_close(g_battery_hid);
        g_battery_hid = NULL;
    }
}

// DualSense USB report 0x01 carries status at byte 53; Bluetooth report 0x31 carries it at byte 54. Capacity is in
// 10% buckets (report values 0..10), and the upper nibble is the charging state (0x1 charging, 0x2 full).
static void dualsense_battery_poll(SDL_Gamepad *pad) {
    if (pad != g_battery_pad || (!g_battery_hid && plat_now_ns() >= g_battery_retry_at)) dualsense_battery_open(pad);
    if (!g_battery_hid) return;
    unsigned char report[128];
    int n = SDL_hid_read(g_battery_hid, report, sizeof report);
    if (n < 0) {
        SDL_hid_close(g_battery_hid);
        g_battery_hid = NULL;
        g_battery_retry_at = plat_now_ns() + 5000000000ull;
        return;
    }
    if (n <= 0) return;
    size_t status_at = report[0] == 0x01 ? 53 : report[0] == 0x31 ? 54 : sizeof report;
    if (status_at >= (size_t)n) return;
    unsigned char status = report[status_at];
    unsigned capacity = status & 0x0f;
    unsigned charging = status >> 4;
    if (capacity > 10 || (charging != 0 && charging != 1 && charging != 2)) return;
    g_battery_percent = charging == 2 ? 100 : (int)(capacity * 10 + 5);
    if (g_battery_percent > 100) g_battery_percent = 100;
    g_battery_charging = charging == 1;
}

void dualsense_init(void) {
    plat_mutex_lock(&g_ds_lock);
    static bool atexit_registered = false;
    if (!atexit_registered) {
        atexit_registered = true;
        atexit(dualsense_cleanup);
    }
    if (g_ds_stream) {
        plat_mutex_unlock(&g_ds_lock);
        return;
    }

    // Must be explicitly enabled in configuration
    if (!cfg_bool("TOTK_DUALSENSE")) {
        plat_mutex_unlock(&g_ds_lock);
        return;
    }

    // Must have a verified physical PS5 DualSense connected
    if (!g_current_pad || !dualsense_is_ps5_gamepad(g_current_pad)) {
        plat_mutex_unlock(&g_ds_lock);
        return;
    }

    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        plat_mutex_unlock(&g_ds_lock);
        return;
    }

    int count = 0;
    SDL_AudioDeviceID *devs = SDL_GetAudioPlaybackDevices(&count);
    for (int i = 0; i < count; i++) {
        const char *name = SDL_GetAudioDeviceName(devs[i]);
        if (name && (strstr(name, "DualSense") || strstr(name, "dualsense") || strstr(name, "Wireless Controller"))) {
            g_ds_device = devs[i];
            rt_log("dualsense: matched audio device '%s' (ID %u)", name, g_ds_device);
            break;
        }
    }
    SDL_free(devs);

    if (g_ds_device) {
        g_ds_channels = 4;
        SDL_AudioSpec spec = {SDL_AUDIO_F32, 4, SAMPLE_RATE};
        g_ds_stream = SDL_OpenAudioDeviceStream(g_ds_device, &spec, NULL, NULL);
        if (!g_ds_stream) {
            g_ds_channels = 2;
            spec.channels = 2;
            g_ds_stream = SDL_OpenAudioDeviceStream(g_ds_device, &spec, NULL, NULL);
        }
        if (g_ds_stream) {
            SDL_ResumeAudioStreamDevice(g_ds_stream);
            g_ds_active = true;
            rt_log("dualsense: %d-channel controller audio stream active on '%s'",
                   g_ds_channels, SDL_GetAudioDeviceName(g_ds_device));
        } else {
            rt_log("dualsense: failed to open audio stream on device %u (%s)", g_ds_device, SDL_GetError());
        }
    }
    plat_mutex_unlock(&g_ds_lock);
}

// Routes the game's actual 48kHz audio into DualSense speaker (high-pass) & voice coils (low-pass)
void dualsense_feed_audio(const float *lr, int frames, int rate) {
    if (!cfg_bool("TOTK_DUALSENSE") || !g_ds_stream) {
        return; // Fast path: zero locks, zero device queries, zero stalls on the render/audio thread
    }

    bool enable_speaker = !cfg_value("TOTK_DUALSENSE_AUDIO") || cfg_bool("TOTK_DUALSENSE_AUDIO");
    bool enable_haptics = !cfg_value("TOTK_DUALSENSE_HAPTICS") || cfg_bool("TOTK_DUALSENSE_HAPTICS");
    if (!enable_speaker && !enable_haptics) return;

    float vol = 1.0f;
    if (cfg_value("TOTK_DUALSENSE_VOL")) {
        vol = (float)cfg_int("TOTK_DUALSENSE_VOL") / 100.0f;
        if (vol < 0.0f) vol = 0.0f;
        if (vol > 2.0f) vol = 2.0f;
    }

    float dt = 1.0f / (rate > 0 ? (float)rate : 48000.0f);

    // High-pass filter for controller speaker: 280 Hz cutoff (keeps Link's bow draw, arrow flight whoosh,
    // sword slashes, shield deflections, and UI chimes crystal clear without rattling the 1-inch speaker)
    float rc_hp = 1.0f / (2.0f * PI * 280.0f);
    float a_hp = rc_hp / (rc_hp + dt);

    // Low-pass filter for controller speaker: 6200 Hz cutoff (removes harsh 1-inch resonance)
    float rc_spk_lp = 1.0f / (2.0f * PI * 6200.0f);
    float a_spk_lp = dt / (rc_spk_lp + dt);

    // Low-pass filter for voice coils: 140 Hz cutoff (translates deep impacts, explosions, shield hits,
    // and strikes into authentic physical tactile vibration in the palms)
    float rc_lp = 1.0f / (2.0f * PI * 140.0f);
    float a_lp = dt / (rc_lp + dt);

    uint64_t now = kern_now_ns();
    uint64_t active_until = atomic_load_explicit(&g_speaker_active_until, memory_order_relaxed);
    bool is_active = (now < active_until) || g_bow_drawn || (now < g_shield_kickback_until);

    // If TOTK_DUALSENSE_AUDIO_MODE is "all", keep always active; default is interactive only!
    const char *audio_mode = cfg_value("TOTK_DUALSENSE_AUDIO_MODE");
    if (audio_mode && !strcmp(audio_mode, "all")) {
        is_active = true;
    }

    float target_gain = is_active ? 1.0f : 0.0f;
    float attack_step = 1.0f - expf(-dt / 0.005f);  // 5ms fast attack
    float release_step = 1.0f - expf(-dt / 0.060f); // 60ms smooth release

    #define STACK_FRAMES 1024
    float out_buf[STACK_FRAMES * 4];
    int chunk = frames > STACK_FRAMES ? STACK_FRAMES : frames;

    for (int k = 0; k < chunk; k++) {
        float xl = lr[2 * k];
        float xr = lr[2 * k + 1];

        // Smooth gate envelope tracking (eliminates clicks/pops and suppresses BGM/ambience)
        float step = (target_gain > g_gate_gain) ? attack_step : release_step;
        g_gate_gain += step * (target_gain - g_gate_gain);
        if (g_gate_gain < 0.0001f) g_gate_gain = 0.0f;

        // Bandpass filtered speaker
        float spk_l = a_hp * (g_hp_y1_l + xl - g_hp_x1_l);
        g_hp_x1_l = xl;
        spk_l = g_spk_lp_y1_l + a_spk_lp * (spk_l - g_spk_lp_y1_l);
        g_spk_lp_y1_l = spk_l;

        float spk_r = a_hp * (g_hp_y1_r + xr - g_hp_x1_r);
        g_hp_x1_r = xr;
        spk_r = g_spk_lp_y1_r + a_spk_lp * (spk_r - g_spk_lp_y1_r);
        g_spk_lp_y1_r = spk_r;

        // Low-pass filtered voice-coil haptics
        float hap_l = g_lp_y1_l + a_lp * (xl - g_lp_y1_l);
        g_lp_y1_l = hap_l;

        float hap_r = g_lp_y1_r + a_lp * (xr - g_lp_y1_r);
        g_lp_y1_r = hap_r;

        // 8.5x pre-gain (+18.6 dB) with studio analog soft-saturation curve (tanhf):
        // Brings whisper-quiet bow creaks and arrow whooshes up to loud, commanding, punchy controller levels
        // while guaranteeing that heavy clangs and impacts smoothly saturate without digital clipping
        float eff_spk_l = tanhf(spk_l * vol * g_gate_gain * 8.5f);
        float eff_spk_r = tanhf(spk_r * vol * g_gate_gain * 8.5f);

        // Physical voice-coil bow recoil punch in the palm actuators (Channels 2 & 3)
        float recoil_kick = 0.0f;
        if (now < g_bow_recoil_until) {
            float elapsed_sec = (float)(140000000ull - (g_bow_recoil_until - now)) / 1000000000.0f;
            if (elapsed_sec >= 0.0f && elapsed_sec < 0.10f) {
                // 65 Hz damped sine shockwave delivering an immediate physical 'THUMP' to palms
                float t = elapsed_sec + (float)k * dt;
                float env = expf(-t * 35.0f);
                recoil_kick = sinf(2.0f * PI * 65.0f * t) * env * 1.0f;
            }
        }

        float eff_hap_l = tanhf(hap_l * 1.8f * g_gate_gain + recoil_kick);
        float eff_hap_r = tanhf(hap_r * 1.8f * g_gate_gain + recoil_kick);

        if (g_ds_channels == 4) {
            out_buf[4 * k + 0] = enable_speaker ? eff_spk_l : 0.0f;
            out_buf[4 * k + 1] = enable_speaker ? eff_spk_r : 0.0f;
            out_buf[4 * k + 2] = enable_haptics ? eff_hap_l : 0.0f;
            out_buf[4 * k + 3] = enable_haptics ? eff_hap_r : 0.0f;
        } else {
            // 2-channel stereo speaker fallback
            out_buf[2 * k + 0] = enable_speaker ? eff_spk_l : 0.0f;
            out_buf[2 * k + 1] = enable_speaker ? eff_spk_r : 0.0f;
        }
    }

    plat_mutex_lock(&g_ds_lock);
    if (g_ds_stream) {
        int queued = SDL_GetAudioStreamQueued(g_ds_stream);
        // Zero-latency synchronization: keep buffer under 40ms
        if (queued > (int)(SAMPLE_RATE * g_ds_channels * sizeof(float) * 0.04f)) {
            SDL_ClearAudioStream(g_ds_stream);
        }
        SDL_PutAudioStreamData(g_ds_stream, out_buf, chunk * g_ds_channels * sizeof(float));
    }
    plat_mutex_unlock(&g_ds_lock);
}

void dualsense_play_sfx(DualSenseSfx sfx) {
    (void)sfx;
    // Real in-game audio from Nintendo's DSP is now continuously streamed to the speaker & haptics!
}

static void send_trigger_effects(SDL_Gamepad *pad, uint8_t r2_mode, uint8_t r2_p0, uint8_t r2_p1, uint8_t r2_p2,
                                 uint8_t l2_mode, uint8_t l2_p0, uint8_t l2_p1, uint8_t l2_p2) {
    if (!pad) return;

    DS5EffectsState_t state;
    memset(&state, 0, sizeof(state));

    state.ucEnableBits1 = 0x04 | 0x08; // bit 2: Right Trigger, bit 3: Left Trigger
    state.ucEnableBits2 = 0x04;        // bit 2: Lightbar LED

    // Zonai Ultrahand / Sage luminous teal-green: #20F8B8 -> R:32, G:248, B:184
    state.ucLedFlags = 0x02;
    state.ucLedBrightness = 0x00;
    state.ucLedRed = 32;
    state.ucLedGreen = 248;
    state.ucLedBlue = 184;

    // Right trigger (Bow)
    state.rgucRightTriggerEffect[0] = r2_mode;
    state.rgucRightTriggerEffect[1] = r2_p0;
    state.rgucRightTriggerEffect[2] = r2_p1;
    state.rgucRightTriggerEffect[3] = r2_p2;

    // Left trigger (Shield)
    state.rgucLeftTriggerEffect[0] = l2_mode;
    state.rgucLeftTriggerEffect[1] = l2_p0;
    state.rgucLeftTriggerEffect[2] = l2_p1;
    state.rgucLeftTriggerEffect[3] = l2_p2;

    SDL_SendGamepadEffect(pad, &state, sizeof(state));
    SDL_SetGamepadLED(pad, 32, 248, 184);
}

void dualsense_poll(void *sdl_gamepad, const uint8_t *pd, int16_t l2_raw, int16_t r2_raw) {
    SDL_Gamepad *pad = (SDL_Gamepad *)sdl_gamepad;
    if (!pad) return;

    // Strict hardware gating: only PS5 DualSense controllers run DualSense routines
    if (!dualsense_is_ps5_gamepad(pad)) {
        if (pad == g_battery_pad) dualsense_battery_close();
        return;
    }
    dualsense_battery_poll(pad);
    if (!cfg_bool("TOTK_DUALSENSE")) return;
    g_current_pad = pad;

    static bool led_initialized = false;
    if (!led_initialized) {
        led_initialized = true;
        // Zonai Ultrahand luminous green lightbar initialization (#20F8B8)
        SDL_SetGamepadLED(pad, 32, 248, 184);
    }

    if (!g_ds_stream) {
        static uint64_t last_init_check = 0;
        uint64_t now_poll = plat_now_ns();
        if (now_poll - last_init_check > 5000000000ull) {
            last_init_check = now_poll;
            dualsense_init();
        }
    }

    uint64_t now = kern_now_ns();

    bool triggers_enabled = (!cfg_value("TOTK_DUALSENSE") || cfg_bool("TOTK_DUALSENSE")) &&
                            (!cfg_value("TOTK_DUALSENSE_TRIGGERS") || cfg_bool("TOTK_DUALSENSE_TRIGGERS"));
    int strength = cfg_value("TOTK_DUALSENSE_TRIGGER_STRENGTH") ? cfg_int("TOTK_DUALSENSE_TRIGGER_STRENGTH") : 100;
    if (strength <= 0) triggers_enabled = false;

    // 1. BOW TRIGGER (R2 / ZR) - Resistance on pull, snappy mechanical recoil on release
    bool r2_pressed = r2_raw > 4000 || (pd && pd[PAD_ZR]);
    uint8_t target_r2_mode = 0;
    uint8_t r2_p0 = 0, r2_p1 = 0, r2_p2 = 0;

    if (r2_pressed) {
        if (!g_bow_drawn) {
            g_bow_drawn = true;
            g_bow_drawn_at = now;
        }
        atomic_store(&g_speaker_active_until, now + 200000000ull); // Keep speaker unmuted while drawing
        if (triggers_enabled) {
            target_r2_mode = 0x02; // Section resistance simulating bowstring draw
            r2_p0 = 25;            // Resistance begins early in draw
            r2_p1 = 190;           // Sustained through pull
            r2_p2 = (uint8_t)((7 * strength + 50) / 100);
            if (r2_p2 < 1) r2_p2 = 1;
        }
    } else {
        if (g_bow_drawn) {
            // Arrow release: fire maximum-power punchy bow recoil jolt & dual-motor snap
            if (g_last_r2 > 10000 && (now - g_bow_drawn_at > 100000000ull)) {
                g_bow_recoil_until = now + 140000000ull; // 140ms bow recoil impulse
                SDL_RumbleGamepad(pad, 65535, 65535, 140); // 100% maximum motor force
                atomic_store(&g_speaker_active_until, now + 900000000ull); // 900ms for arrow whoosh & hit
            }
            g_bow_drawn = false;
        }
        if (now < g_bow_recoil_until && triggers_enabled) {
            // RECOIL KICK: Heavy mechanical hammer against index finger simulating high-tension bowstring snap
            target_r2_mode = 0x06; // Vibration mode
            r2_p0 = 0;
            r2_p1 = 8;             // Maximum motor force (8/8)
            r2_p2 = 30;            // 30 Hz heavy mechanical kick
        } else {
            target_r2_mode = 0x00;
        }
    }

    // 2. SHIELD TRIGGER (L2 / ZL) - Continuous resistance & violent kickback on hit
    bool l2_pressed = l2_raw > 4000 || (pd && pd[PAD_ZL]);
    g_shield_raised = l2_pressed;
    uint8_t target_l2_mode = 0;
    uint8_t l2_p0 = 0, l2_p1 = 0, l2_p2 = 0;

    if (now < g_shield_kickback_until) {
        // ENEMY HIT SHIELD: Violent mechanical kickback hammering against finger!
        target_l2_mode = 0x06; // Vibration mode
        l2_p0 = 0;             // Kick starting immediately
        l2_p1 = 8;             // Maximum motor force
        l2_p2 = 35;            // 35 Hz pulsing hammer frequency
        atomic_store(&g_speaker_active_until, now + 650000000ull);
    } else if (l2_pressed) {
        if (triggers_enabled) {
            target_l2_mode = 0x01; // Continuous resistance
            l2_p0 = 15;            // Start pos
            l2_p1 = (uint8_t)((5 * strength + 50) / 100);
            if (l2_p1 < 1) l2_p1 = 1;
        }
    } else {
        target_l2_mode = 0x00;
    }

    // Send packet to controller only when trigger effect mode or parameters change
    if (target_r2_mode != g_current_r2_mode || target_l2_mode != g_current_l2_mode) {
        send_trigger_effects(pad, target_r2_mode, r2_p0, r2_p1, r2_p2, target_l2_mode, l2_p0, l2_p1, l2_p2);
        g_current_r2_mode = target_r2_mode;
        g_current_l2_mode = target_l2_mode;
    }

    g_last_r2 = r2_raw;
    g_last_l2 = l2_raw;
    (void)g_last_l2;
}

void dualsense_on_vibration(int side, float amp_low, float freq_low, float amp_high, float freq_high) {
    (void)freq_low; (void)freq_high;
    uint64_t now = kern_now_ns();

    // In-game dynamic audio gating: Unmute speaker for 450ms on impacts, melee hits, explosions, chest fanfares
    if (amp_low > 0.08f || amp_high > 0.08f) {
        atomic_store(&g_speaker_active_until, now + 450000000ull);
    }

    // Shield Block: When enemy strikes raised shield, activate 350ms violent trigger kickback
    if (g_shield_raised && side == 0 && (amp_low > 0.14f || amp_high > 0.14f)) {
        if (now - g_last_block_time > 220000000ull) {
            g_last_block_time = now;
            g_shield_kickback_until = now + 350000000ull;
            atomic_store(&g_speaker_active_until, now + 650000000ull);
            if (g_current_pad) {
                // Instantly send trigger kickback packet
                send_trigger_effects(g_current_pad, g_current_r2_mode, 0, 0, 0, 0x06, 0, 8, 35);
                g_current_l2_mode = 0x06;
                SDL_RumbleGamepad(g_current_pad, 45000, 62000, 220);
            }
        }
    }
}

bool dualsense_active(void) {
    return g_ds_active;
}

int dualsense_power_info(void *sdl_gamepad, int *percent, int *charging) {
    SDL_Gamepad *pad = (SDL_Gamepad *)sdl_gamepad;
    if (!pad || pad != g_battery_pad || !dualsense_is_ps5_gamepad(pad) || g_battery_percent < 0) return 0;
    if (percent) *percent = g_battery_percent;
    if (charging) *charging = g_battery_charging;
    return 1;
}

void dualsense_power_reset(void) { dualsense_battery_close(); }

void dualsense_cleanup(void) {
    dualsense_battery_close();
    plat_mutex_lock(&g_ds_lock);
    if (g_current_pad) {
        // 1. Reset adaptive triggers back to normal free mode
        send_trigger_effects(g_current_pad, 0, 0, 0, 0, 0, 0, 0, 0);

        // 2. Turn off Zonai green LED and restore default system lightbar state
        SDL_SetGamepadLED(g_current_pad, 0, 0, 0);
        SDL_RumbleGamepad(g_current_pad, 0, 0, 0);

        DS5EffectsState_t state;
        memset(&state, 0, sizeof(state));
        state.ucEnableBits1 = 0x04 | 0x08; // Triggers reset
        state.ucEnableBits2 = 0x04;        // LED
        state.ucLedFlags = 0x02;           // Set color
        state.ucLedBrightness = 0x00;
        state.ucLedRed = 0;
        state.ucLedGreen = 0;
        state.ucLedBlue = 0;
        SDL_SendGamepadEffect(g_current_pad, &state, sizeof(state));

        // Allow USB HID packet to flush over the wire before process termination
        plat_sleep_ns(20000000);
    }
    if (g_ds_stream) {
        SDL_DestroyAudioStream(g_ds_stream);
        g_ds_stream = NULL;
    }
    g_ds_active = false;
    plat_mutex_unlock(&g_ds_lock);
}
