// The start screen's background (ui/launcher/launcher_backdrop.cpp).
#pragma once
#include <SDL3/SDL.h>

struct ImDrawList;
void backdrop_load(SDL_Renderer *ren, const char *exe_dir);   // splash.png next to the program, if there is one
// The window's flat colour; picture: the splash over all of it (the main page)
void backdrop_draw(ImDrawList *dl, float w, float h, bool picture);
void backdrop_free(void);
bool backdrop_has_picture(void);   // there is a splash.png
SDL_Texture *backdrop_icon(void);  // THWOMP: the MKW icon, after backdrop_load (NULL: none)
