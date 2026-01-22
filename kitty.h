/*
 * kitty.h - Kitty terminal graphics protocol support
 */

#ifndef KITTY_H
#define KITTY_H

#include "flux.h"

/*
 * Display image in terminal using Kitty graphics protocol.
 * Returns 0 on success, -1 on error.
 */
int kitty_display_image(const flux_image *img);

/*
 * Display PNG file in terminal using Kitty graphics protocol.
 * Returns 0 on success, -1 on error.
 */
int kitty_display_png(const char *path);

#endif /* KITTY_H */
