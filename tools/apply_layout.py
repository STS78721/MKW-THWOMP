# One-off: puts the Mario Kart World PC window layout and style into runtime/ui/menu.cpp and its texts into ui_text.
import sys
from pathlib import Path

root = Path(sys.argv[1])
menu = root / 'runtime' / 'ui' / 'menu.cpp'
s = menu.read_text(encoding='utf-8')


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('#include "ui/option_rows.h"\n', '#include "ui/option_rows.h"\n#include "ui/theme.h"\n')
i = s.index('// ---- the window: title, tabs (L / R or Q / E switch)')
j = s.index('static int g_save_pending;')
s = s[:i] + (root / 'tools' / 'new_layout.cpp.txt').read_text(encoding='utf-8') + s[j:]

i = s.index("// SavageWindRecomp's colours: dark green panels, gold accents.\nstatic void style_init(ImGuiStyle &s) {")
j = s.index('static void make_context(void) {')
s = s[:i] + '''// The Mario Kart World PC theme (ui/theme.h): the Switch 2 system settings look, light panels and cyan selection.
static void style_init(ImGuiStyle &s) {
    using namespace mkw_theme;
    ImGui::StyleColorsLight(&s);
    s.WindowRounding = 18; s.FrameRounding = 20; s.GrabRounding = 8; s.ChildRounding = 10; s.PopupRounding = 14;
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
    c[ImGuiCol_ButtonHovered] = ImVec4(0.90f, 0.97f, 0.99f, 1);
    c[ImGuiCol_ButtonActive] = ImVec4(0.78f, 0.93f, 0.97f, 1);
    c[ImGuiCol_NavCursor] = v4(kAccent);
    c[ImGuiCol_Separator] = v4(kDivider);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.72f, 0.72f, 0.75f, 1);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.60f, 0.60f, 0.64f, 1);
    c[ImGuiCol_ScrollbarGrabActive] = v4(kAccent);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.30f);
}

''' + s[j:]
menu.write_text(s, encoding='utf-8', newline='')

# texts: the footer hints and the version line
h = root / 'runtime' / 'ui_text.h'
t = h.read_text(encoding='utf-8')
old = "'UI_MENU_MEMORY_IMPORT_REC', 'UI_MENU_GPU_DESC_RESTART'"
t = t.replace('UI_MENU_MEMORY_IMPORT_REC, UI_MENU_GPU_DESC_RESTART,',
              'UI_MENU_MEMORY_IMPORT_REC, UI_MENU_GPU_DESC_RESTART, UI_MENU_HINT_OK, UI_MENU_HINT_BACK,', 1)
assert 'UI_MENU_HINT_OK' in t
h.write_text(t, encoding='utf-8', newline='')
c = root / 'runtime' / 'ui_text.c'
t = c.read_text(encoding='utf-8')
t = t.replace('    [UI_MENU_TAB_DISPLAY] = "Display",',
              '    [UI_MENU_HINT_OK] = "OK", [UI_MENU_HINT_BACK] = "Back",\n    [UI_MENU_TAB_DISPLAY] = "Display",', 1)
t = t.replace('    [UI_MENU_TAB_DISPLAY] = "Экран",',
              '    [UI_MENU_HINT_OK] = "OK", [UI_MENU_HINT_BACK] = "Назад",\n    [UI_MENU_TAB_DISPLAY] = "Экран",', 1)
t = t.replace('[UI_MENU_VERSION] = "Mario Kart World PC %s, game %s"', '[UI_MENU_VERSION] = "PC %s  ·  update %s"')
t = t.replace('[UI_MENU_VERSION] = "Mario Kart World PC %s, игра %s"', '[UI_MENU_VERSION] = "PC %s  ·  обновление %s"')
assert t.count('UI_MENU_HINT_BACK]') == 2 and 'update %s' in t
c.write_text(t, encoding='utf-8', newline='')
print('ok')
