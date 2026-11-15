// The start screen's look (launcher_look.h): its ImGui style and the colours of the settings rows on its Settings page.
// Only the start screen uses it; the in-game menu keeps menu_look_style and the rows' own colours.
#include "ui/launcher/launcher_look.h"
#include "ui/menu/menu_theme.h"

LookPalette g_look_pal;

// THWOMP: the two palettes (launcher_look.h); dark is the default theme.
static void look_palette(bool dark) {
    if (dark)
        g_look_pal = LookPalette{
            IM_COL32(14, 14, 17, 255), IM_COL32(22, 22, 26, 255), IM_COL32(38, 38, 45, 255), IM_COL32(56, 40, 46, 255),
            IM_COL32(74, 44, 52, 255), IM_COL32(200, 28, 40, 255), IM_COL32(226, 44, 56, 255), IM_COL32(170, 22, 32, 255),
            IM_COL32(238, 238, 242, 255), IM_COL32(186, 186, 196, 255), IM_COL32(150, 150, 162, 255),
            IM_COL32(86, 86, 96, 255), IM_COL32(255, 255, 255, 235), IM_COL32(255, 255, 255, 9),
            IM_COL32(255, 255, 255, 24), IM_COL32(255, 255, 255, 24), IM_COL32(91, 201, 138, 255),
            IM_COL32(236, 160, 70, 255), IM_COL32(244, 112, 100, 255),
            IM_COL32(12, 11, 13, 255), IM_COL32(30, 22, 26, 255), IM_COL32(58, 30, 36, 255), true};
    else
        g_look_pal = LookPalette{
            IM_COL32(232, 232, 236, 255), IM_COL32(244, 244, 246, 255), IM_COL32(255, 255, 255, 255),
            IM_COL32(230, 247, 252, 255), IM_COL32(199, 237, 247, 255), IM_COL32(0, 128, 168, 255),
            IM_COL32(0, 146, 188, 255), IM_COL32(0, 110, 146, 255), IM_COL32(38, 38, 44, 255), IM_COL32(88, 88, 96, 255),
            IM_COL32(104, 104, 112, 255), IM_COL32(176, 176, 182, 255), IM_COL32(0, 176, 214, 255),
            IM_COL32(0, 0, 0, 8), IM_COL32(0, 0, 0, 16), IM_COL32(0, 0, 0, 28), IM_COL32(30, 150, 80, 255),
            IM_COL32(196, 110, 20, 255), IM_COL32(200, 60, 50, 255),
            IM_COL32(214, 214, 220, 255), IM_COL32(236, 236, 240, 255), IM_COL32(250, 250, 252, 255), false};
    launcher_look_rows = OptLook{
        /* select */ LOOK_FOCUS, /* select_off */ (LOOK_FOCUS & ~IM_COL32_A_MASK) | (110u << IM_COL32_A_SHIFT),
        /* divider */ LOOK_LINE, /* dim */ LOOK_DIM, /* off */ LOOK_OFF, /* arrow */ LOOK_DIM,
        /* label */ LOOK_TEXT, /* value */ LOOK_TEXT, /* value_default */ LOOK_HEADING,
        /* note */ LOOK_WARN, /* note_off */ LOOK_WARN, /* fill */ LOOK_ROW, /* fill_hot */ LOOK_ROW_HOT,
        /* border */ LOOK_LINE, /* border_off */ LOOK_ROW, /* cell_off */ LOOK_ROW,
        /* group */ LOOK_HEADING, /* warn */ LOOK_WARN, /* radius */ 3, /* desc_fill */ 0, /* underline */ false};
}

// Every colour ImGui draws by itself is set here, the scrollbar, the text selection and the progress bar too: a default
// left in one corner reads as unfinished.
void launcher_look_style(ImGuiStyle &s) {
    look_palette(menu_theme_dark());
    if (g_look_pal.dark) ImGui::StyleColorsDark(&s);
    else ImGui::StyleColorsLight(&s);
    s.WindowRounding = 4; s.FrameRounding = 3; s.GrabRounding = 3; s.TabRounding = 3; s.ChildRounding = 3;
    s.PopupRounding = 3; s.ScrollbarRounding = 3;
    s.WindowPadding = ImVec2(32, 26); s.FramePadding = ImVec2(18, 9); s.ItemSpacing = ImVec2(14, 12);
    s.WindowBorderSize = 1; s.ScrollbarSize = 12; s.TabBarBorderSize = 1; s.TabBarOverlineSize = 2;
    ImVec4 *c = s.Colors;
    auto rgb = [](ImU32 u) { return ImGui::ColorConvertU32ToFloat4(u); };
    ImVec4 accent = rgb(LOOK_ACCENT), raised = rgb(LOOK_RAISED), hover = rgb(LOOK_HOVER),
           pressed = rgb(g_look_pal.pressed), clear(0, 0, 0, 0);
    c[ImGuiCol_WindowBg] = rgb(LOOK_PANEL);
    c[ImGuiCol_PopupBg] = rgb(LOOK_PANEL);
    c[ImGuiCol_ChildBg] = clear;
    c[ImGuiCol_Border] = rgb(LOOK_LINE);
    c[ImGuiCol_Text] = rgb(LOOK_TEXT);
    c[ImGuiCol_TextDisabled] = rgb(LOOK_DIM);
    c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.45f);
    c[ImGuiCol_Button] = raised;
    c[ImGuiCol_ButtonHovered] = hover;
    c[ImGuiCol_ButtonActive] = pressed;
    c[ImGuiCol_FrameBg] = raised;
    c[ImGuiCol_FrameBgHovered] = hover;
    c[ImGuiCol_FrameBgActive] = pressed;
    c[ImGuiCol_Header] = raised;
    c[ImGuiCol_HeaderHovered] = hover;
    c[ImGuiCol_HeaderActive] = pressed;
    c[ImGuiCol_Tab] = clear;
    c[ImGuiCol_TabHovered] = hover;
    c[ImGuiCol_TabSelected] = raised;
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = clear;
    c[ImGuiCol_TabDimmedSelected] = raised;
    c[ImGuiCol_TabDimmedSelectedOverline] = accent;
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = rgb(LOOK_ACCENT_HOT);
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_ScrollbarBg] = clear;
    c[ImGuiCol_ScrollbarGrab] = hover;
    c[ImGuiCol_ScrollbarGrabHovered] = pressed;
    c[ImGuiCol_ScrollbarGrabActive] = accent;
    c[ImGuiCol_NavCursor] = rgb(LOOK_FOCUS);
    c[ImGuiCol_Separator] = rgb(LOOK_LINE);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, g_look_pal.dark ? 0.60f : 0.30f);
}

// The settings rows in the start screen's palette: filled by look_palette.
OptLook launcher_look_rows;
