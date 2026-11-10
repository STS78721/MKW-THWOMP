// Mario Kart World PC menu theme. Light (default): the look of the Switch 2 system settings, light grey panels and a
// cyan selection. Dark (MKW_MENU_THEME=dark): near-black panels with red accents. Every colour the menu draws comes from
// here (ui/menu.cpp, ui/option_rows.cpp); apply() switches the set, the menu re-applies its ImGui style after it.
#pragma once
#include "imgui.h"

namespace mkw_theme {
inline ImU32 kBackdrop, kPanel, kHeader, kSidebar, kDivider;
inline ImU32 kText, kTextSoft, kTextDim, kTextOff;
inline ImU32 kAccent, kAccentSoft, kAccentText;   // selection outline and arrows; changed values, selected category
inline ImU32 kRowSel, kRowHover, kWarn, kHelp;
inline ImU32 kButtonHover, kButtonActive, kScroll;

inline void apply(bool dark) {
    if (!dark) {
        kBackdrop = IM_COL32(0, 0, 0, 150);
        kPanel = IM_COL32(240, 240, 242, 255);
        kHeader = IM_COL32(250, 250, 251, 255);
        kSidebar = IM_COL32(228, 228, 232, 255);
        kDivider = IM_COL32(206, 206, 212, 255);
        kText = IM_COL32(38, 38, 44, 255);
        kTextSoft = IM_COL32(88, 88, 96, 255);
        kTextDim = IM_COL32(138, 138, 146, 255);
        kTextOff = IM_COL32(176, 176, 182, 255);
        kAccent = IM_COL32(0, 176, 214, 255);
        kAccentSoft = IM_COL32(0, 176, 214, 70);
        kAccentText = IM_COL32(0, 128, 168, 255);
        kRowSel = IM_COL32(255, 255, 255, 255);
        kRowHover = IM_COL32(255, 255, 255, 150);
        kWarn = IM_COL32(206, 92, 20, 255);
        kHelp = IM_COL32(250, 250, 251, 255);
        kButtonHover = IM_COL32(230, 247, 252, 255);
        kButtonActive = IM_COL32(199, 237, 247, 255);
        kScroll = IM_COL32(184, 184, 191, 255);
    } else {
        kBackdrop = IM_COL32(0, 0, 0, 175);
        kPanel = IM_COL32(24, 24, 28, 255);
        kHeader = IM_COL32(32, 32, 38, 255);
        kSidebar = IM_COL32(17, 17, 21, 255);
        kDivider = IM_COL32(56, 56, 64, 255);
        kText = IM_COL32(236, 236, 240, 255);
        kTextSoft = IM_COL32(182, 182, 192, 255);
        kTextDim = IM_COL32(128, 128, 140, 255);
        kTextOff = IM_COL32(86, 86, 96, 255);
        kAccent = IM_COL32(232, 36, 48, 255);
        kAccentSoft = IM_COL32(232, 36, 48, 80);
        kAccentText = IM_COL32(255, 92, 98, 255);
        kRowSel = IM_COL32(44, 44, 52, 255);
        kRowHover = IM_COL32(40, 40, 47, 200);
        kWarn = IM_COL32(255, 156, 64, 255);
        kHelp = IM_COL32(30, 30, 36, 255);
        kButtonHover = IM_COL32(70, 34, 40, 255);
        kButtonActive = IM_COL32(104, 30, 38, 255);
        kScroll = IM_COL32(78, 78, 90, 255);
    }
}

inline ImVec4 v4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }
}  // namespace mkw_theme
