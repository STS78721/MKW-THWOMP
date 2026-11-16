// The settings pages (ui/menu/settings_tabs.cpp): the tab bar and the rows of each tab, for the in-game menu and the start
// screen. ImGui's thread only.
#pragma once
#include <vulkan/vulkan.h>

enum { TAB_SCREEN, TAB_GRAPHICS, TAB_ADV_GRAPHICS, TAB_GAME, TAB_SOUND, TAB_CONTROLS, TAB_MENU, TAB_ADVANCED, TAB_HACKS, TAB_COUNT };

void settings_list_gpus(VkInstance inst);   // once: the Vulkan devices the Screen tab offers (MKW_GPU_INDEX)
void settings_select(int tab);              // that tab at the next layout
void settings_opened(void);                 // the pages were (re)opened: the first row takes the cursor
bool settings_modal(void);                  // a confirmation or a key capture is open: B and Esc are its own
void settings_tabs(float scale, float bottom);   // the tab bar and the shown page; bottom: height kept below it
void settings_tabs_done(void);
// THWOMP: the in-game menu's category list draws the tabs itself; these give it the names and draw the page alone
const char *settings_tab_label(int tab);
int settings_tab_shown(void);              // the tab shown, or the one requested for the next layout
void settings_page(float scale, float bottom);   // L / R steps, the shown page's rows (no tab bar)              // after the window's own B / Esc handling, each frame
const char *settings_present_name(const char *mode);   // the menu's name of a MKW_VK_PRESENT mode; NULL for another
