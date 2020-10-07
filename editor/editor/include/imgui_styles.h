#ifndef __IMGUI_STYLES_H__
#define __IMGUI_STYLES_H__

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

namespace ImGui
{
void StyleColorsDracula(ImGuiStyle *dst = NULL);
void StyleColorsCherry(ImGuiStyle *dst = NULL);
void StyleColorsLightGreen(ImGuiStyle *dst = NULL);
void StyleColorsYellow(ImGuiStyle *dst = NULL);
void StyleColorsGrey(ImGuiStyle *dst = NULL);
void StyleColorsCharcoal(ImGuiStyle *dst = NULL);
void StyleColorsCorporate(ImGuiStyle *dst = NULL);
} // namespace ImGui

#endif