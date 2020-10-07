#ifndef __IMGUI_EXTENSIONS_H__
#define __IMGUI_EXTENSIONS_H__

#include <GL/glew.h>
#include <string>
#include <vector>

#include "core/Guid.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

namespace ImGui
{
void ToggleButton(const char *str_id, bool *v);

bool StampButton(const char *label, bool active);
bool StampButtonEx(const char *label, const ImVec2 &size_arg, ImGuiButtonFlags flags, bool active);

bool BeginDropdown(std::string name, std::vector<std::string> values, int *selection);
void EndDropdown();

bool Combo(const char *label, int *currIndex, std::vector<std::string> &values);

bool Slot(const std::string slotLabel, const std::string slotText, bool slotFillable, bool *slotFilled);
bool ImageSlot(const std::string slotLabel, GLuint texture, bool slotFillable, bool *slotFilled);
} // namespace ImGui

#endif
