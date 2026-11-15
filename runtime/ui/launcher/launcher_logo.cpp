// The start screen's logo and motion (launcher_logo.h), split from launcher.cpp, which keeps the pages.
#include <math.h>

#include "ui/launcher/launcher_logo.h"
#include "ui/launcher/launcher_backdrop.h"

// The same shapes as the .svg: a chequered flag on a pole, on a blue rounded square; each check a quad strip whose edges
// follow the wave.
void launcher_logo(ImDrawList *dl, ImVec2 p, float size) {
    if (SDL_Texture *icon = backdrop_icon()) {   // THWOMP: the MKW icon
        dl->AddImage((ImTextureID)(intptr_t)icon, p, ImVec2(p.x + size, p.y + size));
        return;
    }
    auto at = [&](float x, float y) { return ImVec2(p.x + x * size, p.y + y * size); };
    auto wave = [](float x) { return 0.045f * sinf(x * 2 * 3.14159265f * 1.1f); };
    ImU32 top = IM_COL32(36, 84, 196, 255), bottom = IM_COL32(18, 38, 110, 255);
    // the rounded square with its gradient: the gradient filled square, its corners cut by the background's colour
    // would show on a picture; so a rounded fill, then the gradient clipped inside its straight middle band
    float r = 0.18f;
    dl->AddRectFilled(at(0, 0), at(1, 1), top, r * size);
    dl->AddRectFilled(at(0, 0.5f), at(1, 1), bottom, r * size, ImDrawFlags_RoundCornersBottom);
    dl->AddRectFilledMultiColor(at(0, r), at(1, 1 - r), top, top, bottom, bottom);
    dl->AddRectFilled(at(0.24f, 0.16f), at(0.30f, 0.86f), IM_COL32(235, 235, 240, 255));
    const int kSteps = 6;
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 4; col++) {
            ImU32 c = (row + col) % 2 == 0 ? IM_COL32(250, 250, 250, 255) : IM_COL32(20, 20, 24, 255);
            for (int i = 0; i < kSteps; i++) {
                float f0 = (col + (float)i / kSteps) / 4, f1 = (col + (float)(i + 1) / kSteps) / 4;
                float x0 = 0.30f + 0.52f * f0, x1 = 0.30f + 0.52f * f1, y0 = 0.20f + row * 0.14f, y1 = y0 + 0.14f;
                dl->AddQuadFilled(at(x0, y0 + wave(x0)), at(x1 + 0.002f, y0 + wave(x1)), at(x1 + 0.002f, y1 + wave(x1)),
                                  at(x0, y1 + wave(x0)), c);
            }
        }
}

float launcher_ease_out(float p) {
    p = fminf(1.0f, fmaxf(0.0f, p));
    return 1 - (1 - p) * (1 - p) * (1 - p);
}

// As Dusklight's start screen does: the elements come in one after the other.
float launcher_intro(double t0, int i) {
    if (t0 <= 0) return 1;
    return launcher_ease_out((float)((ImGui::GetTime() - t0 - 0.08 * i) / 0.35));
}

// One call fades whatever an element drew, its shapes and its text alike. The hit box stays put.
void launcher_intro_apply(ImDrawList *dl, int start, float k, float scale) {
    if (k >= 1) return;
    float dy = (1 - k) * 12 * scale;
    for (int i = start; i < dl->VtxBuffer.Size; i++) {
        ImDrawVert &v = dl->VtxBuffer[i];
        v.pos.y += dy;
        ImU32 a = (v.col >> IM_COL32_A_SHIFT) & 0xFF;
        v.col = (v.col & ~IM_COL32_A_MASK) | ((ImU32)(a * k) << IM_COL32_A_SHIFT);
    }
}

bool launcher_fade(ImDrawList *dl, float w, float h, ImU32 c, double t0, float secs) {
    float k = launcher_ease_out((float)((ImGui::GetTime() - t0) / secs));
    if (k >= 1) return false;
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), (c & ~IM_COL32_A_MASK) | ((ImU32)(255 * (1 - k)) << IM_COL32_A_SHIFT));
    return true;
}
