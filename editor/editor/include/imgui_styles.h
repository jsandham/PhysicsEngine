#ifndef __IMGUI_STYLES_H__
#define __IMGUI_STYLES_H__

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

namespace ImGui
{
	void	StyleColorsDracula(ImGuiStyle* dst = NULL);
	void	StyleColorsCherry(ImGuiStyle* dst = NULL);
	void	StyleColorsLightGreen(ImGuiStyle* dst = NULL);
	void	StyleColorsYellow(ImGuiStyle* dst = NULL);
	void	StyleColorsGrey(ImGuiStyle* dst = NULL);
	void	StyleColorsCharcoal(ImGuiStyle* dst = NULL);
	void	StyleColorsCorporate(ImGuiStyle* dst = NULL);
}

#endif