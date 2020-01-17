#ifndef __IMGUI_EXTENSIONS_H__
#define __IMGUI_EXTENSIONS_H__

#include <vector>
#include <string>

#include "core/Guid.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

namespace ImGui
{
	bool BeginDropdown(std::string name, std::vector<std::string> values, int* selection);
	void EndDropdown();

	/*auto vector_getter = [](void* vec, int idx, const char** out_text);*/
	bool Combo(const char* label, int* currIndex, std::vector<std::string>& values);

	bool Slot(const std::string slotLabel, const std::string slotText, bool slotFillable, bool* slotFilled);
}

#endif
