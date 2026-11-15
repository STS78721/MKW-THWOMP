// The look the in-game menu (ui/menu/menu.cpp) and the start screen (ui/launcher/launcher.cpp) share, split from menu.cpp. The colours
// follow SavageWindRecomp's runtime/ui/menu.mm (https://github.com/Revivedx/SavageWindRecomp @
// 575ad88e02f9a89a9a885d90039d48af33c982f4, MIT, Copyright (c) 2026 Revivedx and the Savage Wind Recomp contributors,
// Copyright (c) 2026 Peiikai).
#include <stdlib.h>
#include <string.h>

#include "ui/menu/menu_look.h"
#include "ui/menu/menu_theme.h"
#include "ui/menu/option_rows.h"

extern "C" {
#include "cfg.h"
#include "platform/platform.h"
#include "ui_text.h"
void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
}

// Roboto Medium (Apache-2.0, ui/fonts/LICENSE-Roboto.txt) with Latin and Cyrillic glyphs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"   // #embed in C++
// The path is relative to this file: #embed does not search the -I directories, and "ui/fonts/..." embedded nothing
// after the move to ui/menu/ (23c2d01): the button icons parsed an empty font and every run fell at the title.
static const unsigned char roboto_ttf[] = {
#embed "../fonts/Roboto-Medium.ttf"
};
#pragma clang diagnostic pop
static_assert(sizeof roboto_ttf > 100000, "Roboto-Medium.ttf not embedded");

extern "C" const unsigned char *menu_look_font_data(void) { return roboto_ttf; }   // ui/glyphs/glyphs.c

float menu_look_scale(uint32_t h) { return (h >= 720 ? h : 720) / 1080.0f; }

int menu_theme_dark(void) { return strcmp(cfg_value("MKW_MENU_THEME"), "light") != 0; }   // dark and SteelTemple
int menu_theme_steeltemple(void) { return !strcmp(cfg_value("MKW_MENU_THEME"), "steeltemple"); }

// THWOMP: the Mario Kart World PC theme (ui/menu/menu_theme.h): light, the Switch 2 system settings with a cyan
// selection, or dark with red accents (MKW_MENU_THEME). The menu re-runs this when the setting changes.
void menu_look_style(ImGuiStyle &s) {
    using namespace mkw_theme;
    bool dark = menu_theme_dark();
    apply(dark);
    opt_menu_look_refresh();
    if (dark) ImGui::StyleColorsDark(&s);
    else ImGui::StyleColorsLight(&s);
    s.WindowRounding = 18; s.FrameRounding = 20; s.GrabRounding = 8; s.TabRounding = 8; s.ChildRounding = 10;
    s.PopupRounding = 14;
    s.WindowPadding = ImVec2(28, 22); s.FramePadding = ImVec2(18, 9); s.ItemSpacing = ImVec2(14, 12);
    s.WindowBorderSize = 0; s.PopupBorderSize = 1; s.FrameBorderSize = 1; s.ScrollbarSize = 14; s.ScrollbarRounding = 7;
    ImVec4 *c = s.Colors;
    c[ImGuiCol_WindowBg] = v4(kPanel);
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = v4(kHeader);
    c[ImGuiCol_Border] = v4(kDivider);
    c[ImGuiCol_Text] = v4(kText);
    c[ImGuiCol_TextDisabled] = v4(kTextDim);
    c[ImGuiCol_FrameBg] = v4(kRowSel);
    c[ImGuiCol_Button] = v4(kRowSel);
    c[ImGuiCol_ButtonHovered] = v4(kButtonHover);
    c[ImGuiCol_ButtonActive] = v4(kButtonActive);
    c[ImGuiCol_Tab] = v4(kSidebar);
    c[ImGuiCol_TabHovered] = v4(kRowHover);
    c[ImGuiCol_TabSelected] = v4(kRowSel);
    c[ImGuiCol_TabSelectedOverline] = v4(kAccent);
    c[ImGuiCol_NavCursor] = v4(kAccent);
    c[ImGuiCol_Separator] = v4(kDivider);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = v4(kScroll);
    c[ImGuiCol_ScrollbarGrabHovered] = v4(kTextDim);
    c[ImGuiCol_ScrollbarGrabActive] = v4(kAccent);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.30f);
}

// Roboto Medium (Latin with Latin Extended-A for Polish, Turkish and Catalan, Cyrillic, dashes and quotes) at 26 px
// per 1080 lines, with the glyphs of the Japanese, Korean and Chinese language names merged in from a system font when
// there is one (each language is named in its own script).
static const ImWchar kRanges[] = {0x0020, 0x017F, 0x0400, 0x052F, 0x2010, 0x2027, 0};
static const struct { const char *lang; UiTextId name; } kCjk[] = {
    {"ja", UI_MENU_LANGNAME_JA}, {"ko", UI_MENU_LANGNAME_KO}, {"zh-Hans", UI_MENU_LANGNAME_ZH_HANS},
    {"zh-Hant", UI_MENU_LANGNAME_ZH_HANT}, {"th", UI_MENU_LANGNAME_TH},
};

static const char *g_font_lang;                        // the menu language the atlas was built for
static ImVector<ImWchar> g_ranges[sizeof kCjk / sizeof *kCjk];   // read when the texture is built, after this returns

int menu_look_fonts_stale(void) { return g_font_lang != menu_lang(); }

// A CJK menu language takes the glyphs of its whole catalog from the system font; the other rows only the names of
// the CJK languages in the language lists. Without the system font the CJK menu shows boxes.
void menu_look_fonts(float scale) {
    enum { N = sizeof kCjk / sizeof *kCjk };
    g_font_lang = menu_lang();
    ImFontAtlas *fa = ImGui::GetIO().Fonts;
    fa->Clear();
    ImFontConfig fc;
    fc.FontDataOwnedByAtlas = false;
    fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, 26.0f * scale, &fc, kRanges);
    int merged = 0;
    for (int i = 0; i < N; i++) {
        char path[512];
        uint64_t size = 0;
        uint8_t *data;
        if (!plat_font_path(kCjk[i].lang, path, sizeof path) || !(data = plat_file_read(path, &size))) continue;
        merged++;
        ImFontGlyphRangesBuilder b;
        b.AddText(ui_text_in("en", kCjk[i].name));
        if (!strcmp(kCjk[i].lang, g_font_lang))
            for (int id = 0; id < UI_TEXT_COUNT; id++) b.AddText(ui_text_in(g_font_lang, (UiTextId)id));
        g_ranges[i].clear();
        b.BuildRanges(&g_ranges[i]);
        ImFontConfig mc;
        mc.MergeMode = true;   // the atlas frees data (IM_FREE is free by default)
        fa->AddFontFromMemoryTTF(data, (int)size, 26.0f * scale, &mc, g_ranges[i].Data);
    }
    rt_log("menu: font atlas at %.1f px for menu language %s, CJK glyphs from %d of %d system fonts", 26.0f * scale,
           g_font_lang, merged, (int)N);
}

// The launcher's other sizes: Roboto, with the system font of a CJK or Thai menu language merged in like the main
// font, or its titles and buttons show "?" for every glyph (discussion #193). Call after menu_look_fonts.
ImFont *menu_look_font_at(float px) {
    ImFontAtlas *fa = ImGui::GetIO().Fonts;
    ImFontConfig fc;
    fc.FontDataOwnedByAtlas = false;
    ImFont *font = fa->AddFontFromMemoryTTF((void *)roboto_ttf, (int)sizeof roboto_ttf, px, &fc, kRanges);
    for (size_t i = 0; i < sizeof kCjk / sizeof *kCjk; i++) {
        char path[512];
        uint64_t size = 0;
        uint8_t *data;
        if (!g_font_lang || strcmp(kCjk[i].lang, g_font_lang) || g_ranges[i].empty() ||
            !plat_font_path(kCjk[i].lang, path, sizeof path) || !(data = plat_file_read(path, &size)))
            continue;
        ImFontConfig mc;
        mc.MergeMode = true;   // the atlas frees data
        fa->AddFontFromMemoryTTF(data, (int)size, px, &mc, g_ranges[i].Data);
    }
    return font;
}
