# One-off: restyles runtime/ui/option_rows.cpp to the Mario Kart World PC theme (ui/theme.h).
import sys

fn = sys.argv[1]
s = open(fn, encoding='utf-8').read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('// Two-pane settings pages (option_rows.h): rows on the left, the selected or hovered row\'s description on the right.',
    '// Settings pages (option_rows.h): the rows, and under them a help box with the selected or hovered row\'s description\n'
    '// (Mario Kart World PC theme, ui/theme.h: the Switch 2 system settings look).')
rep('#include "ui/option_rows.h"\n', '#include "ui/option_rows.h"\n#include "ui/theme.h"\n')
rep('''static const ImU32 kGold = IM_COL32(236, 194, 84, 255);
static const ImU32 kDim = IM_COL32(150, 150, 140, 255);
static const ImU32 kGrey = IM_COL32(110, 110, 102, 255);
''', '''using namespace mkw_theme;
static const ImU32 kGold = kAccent;     // the selection
static const ImU32 kDim = kTextDim;
static const ImU32 kGrey = kTextOff;
''')
rep('static float g_bottom;                   // opt_reserve_bottom: kept free under the panes\n',
    'static float g_bottom;                   // opt_reserve_bottom: kept free under the page\n'
    'static float g_right;                    // opt_reserve_right: kept free right of the page\n')
rep('void opt_reserve_bottom(float h) { g_bottom = h; }\n',
    'void opt_reserve_bottom(float h) { g_bottom = h; }\n\nvoid opt_reserve_right(float w) { g_right = w; }\n')

# opt_begin / opt_end: rows on top, a help box under them
rep('''    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= g_bottom;
    ImGui::PushID(id);
    ImGui::BeginChild("##rows", ImVec2(avail.x * 0.58f, avail.y), ImGuiChildFlags_NavFlattened);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 5 * scale));''',
    '''    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= g_bottom + help_height();
    avail.x -= g_right;
    ImGui::PushID(id);
    ImGui::BeginChild("##rows", ImVec2(avail.x, avail.y), ImGuiChildFlags_NavFlattened);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0));''')
rep('void opt_begin(const char *id, float scale) {',
    '''// The help box under the rows: room for about four lines of the description.
static float help_height(void) { return ImGui::GetFontSize() * 5.2f; }

void opt_begin(const char *id, float scale) {''')
i = s.index('    ImGui::SameLine(0, 0);\n    ImVec2 p = ImGui::GetCursorScreenPos(), avail = ImGui::GetContentRegionAvail();')
j = s.index('    ImGui::PopID();\n}\n', i)
s = s[:i] + '''    // the help box: a white rounded box under the rows, the description wrapped in it
    float gap = 12 * g_scale, hh = help_height() - gap;
    ImVec2 p = ImGui::GetCursorScreenPos();
    p.y += gap;
    float w = ImGui::GetContentRegionAvail().x - g_right;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + hh), kHelp, 10 * g_scale);
    dl->AddRect(p, ImVec2(p.x + w, p.y + hh), kDivider, 10 * g_scale);
    float pad = 16 * g_scale;
    ImGui::SetCursorScreenPos(ImVec2(p.x + pad, p.y + pad * 0.7f));
    ImGui::PushStyleColor(ImGuiCol_Text, v4(kTextSoft));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + w - pad * 2);
    ImGui::TextUnformatted(g_shown.text);
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + hh));
''' + s[j:]

# the row: no box; a divider under it, the selected row white with a cyan outline
rep('''    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x, h = floorf(fs * 2.2f);''',
    '''    float fs = ImGui::GetFontSize(), w = ImGui::GetContentRegionAvail().x, h = floorf(fs * 2.35f);''')
rep('''    float r = 9 * g_scale, pad = 16 * g_scale, small = fs * 0.72f;
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), hov || foc ? IM_COL32(255, 255, 255, 22) : IM_COL32(0, 0, 0, 40), r);
    dl->AddRect(p, ImVec2(p.x + w, p.y + h),
                hov || foc ? (off ? IM_COL32(236, 194, 84, 120) : kGold) : IM_COL32(255, 255, 255, off ? 18 : 38), r, 0,
                hov || foc ? 2.0f : 1.0f);''',
    '''    float r = 8 * g_scale, pad = 18 * g_scale, small = fs * 0.72f;
    if (foc) {
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), kRowSel, r);
        dl->AddRect(ImVec2(p.x + 1, p.y + 1), ImVec2(p.x + w - 1, p.y + h - 1), off ? kAccentSoft : kAccent, r, 0,
                    3.0f * g_scale);
    } else {
        if (hov) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), kRowHover, r);
        dl->AddLine(ImVec2(p.x + pad * 0.5f, p.y + h - 0.5f), ImVec2(p.x + w - pad * 0.5f, p.y + h - 0.5f), kDivider, 1.0f);
    }''')
rep('''        ImU32 ac = off ? kGrey : hov || foc ? kGold : IM_COL32(170, 170, 160, 255);''',
    '''        ImU32 ac = off ? kGrey : hov || foc ? kGold : kTextDim;''')
rep('''        dl->AddText(ImVec2(vx, ty), off ? kGrey : v.is_default ? IM_COL32(205, 205, 195, 255) : IM_COL32(255, 255, 255, 255),
                    v.value);''', '''        dl->AddText(ImVec2(vx, ty), off ? kGrey : v.is_default ? kText : kAccentText, v.value);''')
rep('''        dl->AddText(font, small, ImVec2(right - tw, ny), v.note ? IM_COL32(220, 170, 100, 255) : kDim, t);''',
    '''        dl->AddText(font, small, ImVec2(right - tw, ny), v.note ? kWarn : kDim, t);''')
rep('''    dl->AddText(font, fs, ImVec2(p.x + pad, ty), off ? kGrey : IM_COL32(240, 236, 224, 255), menu_text(label), NULL, 0,''',
    '''    dl->AddText(font, fs, ImVec2(p.x + pad, ty), off ? kGrey : kText, menu_text(label), NULL, 0,''')
rep('''    if (note) dl->AddText(font, small, ImVec2(p.x + pad, ny), off ? IM_COL32(200, 150, 90, 255) : kDim, note);''',
    '''    if (note) dl->AddText(font, small, ImVec2(p.x + pad, ny), off ? kWarn : kDim, note);''')

# headings and page text
rep('''    ImGui::Dummy(ImVec2(0, 6 * g_scale));
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", menu_text(title));''',
    '''    ImGui::Dummy(ImVec2(0, 10 * g_scale));
    ImGui::TextColored(v4(kTextSoft), "%s", menu_text(title));
    ImGui::Dummy(ImVec2(0, 2 * g_scale));''')
rep('''    ImGui::Dummy(ImVec2(0, 6 * g_scale));
    ImGui::TextColored(ImVec4(0.93f, 0.76f, 0.33f, 1), "%s", title);''',
    '''    ImGui::Dummy(ImVec2(0, 10 * g_scale));
    ImGui::TextColored(v4(kTextSoft), "%s", title);
    ImGui::Dummy(ImVec2(0, 2 * g_scale));''')
rep('''    if (warn) ImGui::TextColored(ImVec4(0.95f, 0.66f, 0.40f, 1), "%s", menu_text(text));
    else ImGui::TextUnformatted(menu_text(text));''',
    '''    if (warn) ImGui::TextColored(v4(kWarn), "%s", menu_text(text));
    else ImGui::TextColored(v4(kTextSoft), "%s", menu_text(text));
    ImGui::Dummy(ImVec2(0, 6 * g_scale));''')

# button-mapping cells
s = s.replace('IM_COL32(240, 236, 224, 255)', 'kText')
s = s.replace('dl->AddRect(a, b, IM_COL32(255, 255, 255, 14), r);', 'dl->AddRect(a, b, kDivider, r);')
rep('''        dl->AddRectFilled(a, b, hov || foc ? IM_COL32(255, 255, 255, 22) : IM_COL32(0, 0, 0, 40), r);
        dl->AddRect(a, b, hov || foc ? kGold : IM_COL32(255, 255, 255, 38), r, 0, hov || foc ? 2.0f : 1.0f);
        fit_text(dl, a.x + pad, b.x - pad, a.y, h, IM_COL32(255, 255, 255, 255), t, true);''',
    '''        dl->AddRectFilled(a, b, hov || foc ? kRowSel : kRowHover, r);
        dl->AddRect(a, b, hov || foc ? kGold : kDivider, r, 0, hov || foc ? 2.5f * g_scale : 1.0f);
        fit_text(dl, a.x + pad, b.x - pad, a.y, h, kText, t, true);''')
left = [x for x in ('IM_COL32(236', 'IM_COL32(255, 255, 255', 'ImVec4(0.93f') if x in s]
open(fn, 'w', encoding='utf-8', newline='').write(s)
print('ok; leftover old colours:', left)
