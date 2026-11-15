// The start screen's own look (ui/launcher/launcher_look.cpp): its palette, its ImGui style and the colours it gives the
// settings rows it shares with the in-game menu (ui/menu/option_rows.h). The in-game menu keeps its own (menu_look.h).
// ImGui's thread only.
#pragma once
#include "imgui.h"
#include "ui/menu/option_rows.h"

// THWOMP: the palette follows the menu theme (MKW_MENU_THEME, ui/menu/menu_theme.h): dark by default, near-black with
// the red of the MKW icon for the one main action of a page, the selected tab and progress; or light, the Switch 2
// system settings' greys with cyan. The selection is a border in LOOK_FOCUS, so a page never shows two accents competing.
// Dark: white on LOOK_ACCENT 5.3:1; LOOK_TEXT on LOOK_PANEL 15:1, LOOK_DIM 6.4:1. Light: white on LOOK_ACCENT 4.8:1,
// LOOK_TEXT on LOOK_PANEL 13:1, LOOK_DIM 4.9:1 (WCAG wants 4.5:1).
struct LookPalette {
    ImU32 bg, panel, raised, hover, pressed, accent, accent_hot, accent_down, text, heading, dim, off, focus, row,
        row_hot, line, ok, warn, bad;
    ImU32 check_dark, check_lit_lo, check_lit_hi;   // the backdrop's cloth: dark checks, lit checks in and out of a fold
    bool dark;
};
extern LookPalette g_look_pal;   // set by launcher_look_style
#define LOOK_BG (g_look_pal.bg)               // the window behind everything
#define LOOK_PANEL (g_look_pal.panel)         // a page, a bar
#define LOOK_RAISED (g_look_pal.raised)       // a button, a field
#define LOOK_HOVER (g_look_pal.hover)
#define LOOK_ACCENT (g_look_pal.accent)
#define LOOK_ACCENT_HOT (g_look_pal.accent_hot)
#define LOOK_ACCENT_DOWN (g_look_pal.accent_down)
#define LOOK_TEXT (g_look_pal.text)
#define LOOK_HEADING (g_look_pal.heading)
#define LOOK_DIM (g_look_pal.dim)
#define LOOK_OFF (g_look_pal.off)             // a row or a button that takes no input
#define LOOK_FOCUS (g_look_pal.focus)
#define LOOK_ROW (g_look_pal.row)             // a row's fill and its border, at rest and under the cursor
#define LOOK_ROW_HOT (g_look_pal.row_hot)
#define LOOK_LINE (g_look_pal.line)
#define LOOK_OK (g_look_pal.ok)
#define LOOK_WARN (g_look_pal.warn)
#define LOOK_BAD (g_look_pal.bad)

void launcher_look_style(ImGuiStyle &s);   // at scale 1: the caller ScaleAllSizes
extern OptLook launcher_look_rows;   // for opt_look: the settings rows in this palette
