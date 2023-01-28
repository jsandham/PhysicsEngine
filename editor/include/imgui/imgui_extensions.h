#ifndef __IMGUI_EXTENSIONS_H__
#define __IMGUI_EXTENSIONS_H__

#include <string>
#include <vector>

#include "core/Guid.h"

#include "imgui.h"
#include "imgui_internal.h"

namespace ImGui
{
void TextCentered(const ImVec4 &col, const std::string text);

void ToggleButton(const char *str_id, bool *v);

bool StampButton(const char *label, bool active);
bool StampButtonEx(const char *label, const ImVec2 &size_arg, ImGuiButtonFlags flags, bool active);

bool BeginDropdown(const std::string &name, const std::vector<std::string> &values, int *selection);
void EndDropdown();

bool BeginDropdownWindow(const std::string &name, const std::vector<std::string> &values, size_t* index);
void EndDropdownWindow();

bool Combo(const char *label, int *currIndex, std::vector<std::string> &values);

//struct SlotData
//{
//    bool isHovered;
//    bool isClicked;
//    bool releaseTriggered;
//    bool clearClicked;
//
//    SlotData()
//    {
//        isHovered = false;
//        isClicked = false;
//        releaseTriggered = false;
//        clearClicked = false;
//    }
//};
//
//bool Slot(const std::string slotLabel, const std::string slotText, bool* releaseTriggered, bool* clearClicked);
//bool Slot2(const std::string slotLabel, const std::string slotText, SlotData* data);
//bool ImageSlot(const std::string slotLabel, GLuint texture, bool* releaseTriggered, bool* clearClicked);
//bool ImageSlot2(const std::string slotLabel, GLuint texture, SlotData* data);


bool SelectableInput(const char *str_id, bool selected, bool *edited, ImGuiSelectableFlags flags, char *buf,
                     size_t buf_size);

bool Splitter(bool split_vertically, float thickness, float *size1, float *size2, float min_size1, float min_size2,
              float splitter_long_axis_size = -1.0f);
} // namespace ImGui

#endif
