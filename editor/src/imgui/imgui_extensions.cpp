#include <algorithm>
#include <iostream>

#include "../../include/imgui/imgui_extensions.h"
//#include "../include/imgui.h"
//#ifndef IMGUI_DEFINE_MATH_OPERATORS
//#define IMGUI_DEFINE_MATH_OPERATORS
//#endif

namespace ImGui
{
    static inline ImVec2 operator*(const ImVec2& lhs, const float rhs)
    {
        return ImVec2(lhs.x * rhs, lhs.y * rhs);
    }
    static inline ImVec2 operator/(const ImVec2& lhs, const float rhs)
    {
        return ImVec2(lhs.x / rhs, lhs.y / rhs);
    }
    static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs)
    {
        return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);
    }
    static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)
    {
        return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y);
    }
    static inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs)
    {
        return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y);
    }
    static inline ImVec2 operator/(const ImVec2& lhs, const ImVec2& rhs)
    {
        return ImVec2(lhs.x / rhs.x, lhs.y / rhs.y);
    }
    static inline ImVec2& operator+=(ImVec2& lhs, const ImVec2& rhs)
    {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        return lhs;
    }
    static inline ImVec2& operator-=(ImVec2& lhs, const ImVec2& rhs)
    {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        return lhs;
    }
    static inline ImVec2& operator*=(ImVec2& lhs, const float rhs)
    {
        lhs.x *= rhs;
        lhs.y *= rhs;
        return lhs;
    }
    static inline ImVec2& operator/=(ImVec2& lhs, const float rhs)
    {
        lhs.x /= rhs;
        lhs.y /= rhs;
        return lhs;
    }
    static inline ImVec4 operator+(const ImVec4& lhs, const ImVec4& rhs)
    {
        return ImVec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }
    static inline ImVec4 operator-(const ImVec4& lhs, const ImVec4& rhs)
    {
        return ImVec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }
    static inline ImVec4 operator*(const ImVec4& lhs, const ImVec4& rhs)
    {
        return ImVec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
    }

}
using namespace ImGui;

void ImGui::TextCentered(const ImVec4 &col, const std::string text)
{
    float font_size = ImGui::GetFontSize() * text.size() / 2;
    ImGui::SameLine(ImGui::GetWindowSize().x / 2 - font_size + (font_size / 2));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), text.c_str());
}

void ImGui::ToggleButton(const char *str_id, bool *v)
{
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList *draw_list = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f;

    ImGui::InvisibleButton(str_id, ImVec2(width, height));
    if (ImGui::IsItemClicked())
        *v = !*v;

    float t = *v ? 1.0f : 0.0f;

    ImGuiContext &g = *GImGui;
    float ANIM_SPEED = 0.08f;
    if (g.LastActiveId == g.CurrentWindow->GetID(str_id)) // && g.LastActiveIdTimer < ANIM_SPEED)
    {
        float t_anim = ImSaturate(g.LastActiveIdTimer / ANIM_SPEED);
        t = *v ? (t_anim) : (1.0f - t_anim);
    }

    ImU32 col_bg;
    if (ImGui::IsItemHovered())
        col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), ImVec4(0.64f, 0.83f, 0.34f, 1.0f), t));
    else
        col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), ImVec4(0.56f, 0.83f, 0.26f, 1.0f), t));

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
    draw_list->AddCircleFilled(ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f,
                               IM_COL32(255, 255, 255, 255));
}

bool ImGui::StampButtonEx(const char *label, const ImVec2 &size_arg, ImGuiButtonFlags flags, bool active)
{
    ImGuiWindow *window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext &g = *GImGui;
    const ImGuiStyle &style = g.Style;
    const ImGuiID id = window->GetID(label);
    const ImVec2 label_size = CalcTextSize(label, NULL, true);

    ImVec2 pos = window->DC.CursorPos;
    if ((flags & ImGuiButtonFlags_AlignTextBaseLine) &&
        style.FramePadding.y <
            window->DC.CurrLineTextBaseOffset) // Try to vertically align buttons that are smaller/have no padding so
                                               // that text baseline matches (bit hacky, since it shouldn't be a flag)
        pos.y += window->DC.CurrLineTextBaseOffset - style.FramePadding.y;
    ImVec2 size =
        CalcItemSize(size_arg, label_size.x + style.FramePadding.x * 2.0f, label_size.y + style.FramePadding.y * 2.0f);

    const ImRect bb(pos, pos + size);
    ItemSize(size, style.FramePadding.y);
    if (!ItemAdd(bb, id))
        return false;

    //if (window->DC.ItemFlags & ImGuiItemFlags_ButtonRepeat)
    //    flags |= ImGuiButtonFlags_Repeat;
    bool hovered, held;
    bool pressed = ButtonBehavior(bb, id, &hovered, &held, flags);

    // Render
    const ImU32 col = GetColorU32((active)  ? ImGuiCol_ButtonActive
                                  : hovered ? ImGuiCol_ButtonHovered
                                            : ImGuiCol_Button);
    RenderNavHighlight(bb, id);
    RenderFrame(bb.Min, bb.Max, col, true, style.FrameRounding);
    RenderTextClipped(bb.Min + style.FramePadding, bb.Max - style.FramePadding, label, NULL, &label_size,
                      style.ButtonTextAlign, &bb);

    IMGUI_TEST_ENGINE_ITEM_INFO(id, label, window->DC.LastItemStatusFlags);
    return pressed;
}

bool ImGui::StampButton(const char *label, bool active)
{
    return StampButtonEx(label, ImVec2(0, 0), ImGuiButtonFlags_None, active);
}

bool ImGui::BeginDropdown(const std::string &name, const std::vector<std::string> &values, int *selection)
{
    ImGui::SameLine(0.f, 0.f);
    ImGui::PushID(("##" + name).c_str());

    bool pressed = ImGui::Button(&name[0]);
    ImGui::PopID();

    if (pressed)
    {
        ImGui::OpenPopup(("##" + name).c_str());
    }

    if (ImGui::BeginPopup(("##" + name).c_str()))
    {
        std::vector<const char *> temp(values.size());
        for (size_t i = 0; i < values.size(); i++)
        {
            temp[i] = values[i].c_str();
        }
        if (ImGui::ListBox(("##" + name).c_str(), selection, &temp[0], (int)temp.size(), 4))
        {
            ImGui::CloseCurrentPopup();
        }
        return true;
    }

    return false;
}

void ImGui::EndDropdown()
{
    ImGui::EndPopup();
}

bool ImGui::BeginDropdownWindow(const std::string& name, const std::vector<std::string>& values, size_t* index)
{
    ImGui::PushID("##Dropdown");
    bool pressed = ImGui::Button(name.c_str());
    ImGui::PopID();

    if (pressed)
    {
        ImGui::OpenPopup("##Dropdown");
    }

    if (ImGui::BeginPopup("##Dropdown"))
    {
        static char inputBuffer[128];
        ImGui::InputTextWithHint("##Search string", "search...", &inputBuffer[0], IM_ARRAYSIZE(inputBuffer));

        ImGuiTextFilter componentFilter(&inputBuffer[0]);
        std::vector<const char*> filteredComponents;
        std::vector<size_t> filteredIndices;
        for (size_t i = 0; i < values.size(); i++)
        {
            if (componentFilter.PassFilter(values[i].c_str()))
            {
                filteredComponents.push_back(values[i].c_str());
                filteredIndices.push_back(i);
            }
        }

        int s = 0;
        if (ImGui::ListBox("##Filter", &s, filteredComponents.data(), (int)filteredComponents.size()))
        {
            *index = filteredIndices[s];
            ImGui::CloseCurrentPopup();
        }
        return true;
    }

    return false;
}

void ImGui::EndDropdownWindow()
{
    ImGui::EndPopup();
}

static auto vector_getter = [](void *vec, int idx, const char **out_text) {
    auto &vector = *static_cast<std::vector<std::string> *>(vec);
    if (idx < 0 || idx >= static_cast<int>(vector.size()))
    {
        return false;
    }
    *out_text = vector.at(idx).c_str();
    return true;
};

bool ImGui::Combo(const char *label, int *currIndex, std::vector<std::string> &values)
{
    if (values.empty())
    {
        return false;
    }
    return Combo(label, currIndex, vector_getter, static_cast<void *>(&values), (int)values.size());
}

//bool ImGui::Slot(const std::string slotLabel, const std::string slotText, bool* releaseTriggered, bool* clearClicked)
//{
//    ImVec2 windowSize = ImGui::GetWindowSize();
//    windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);
//
//    /*ImGui::ButtonEx(slotText.c_str(), ImVec2(windowSize.x, 0), ImGuiButtonFlags_Disabled);*/
//    ImGui::ButtonEx(slotText.c_str(), ImVec2(windowSize.x, 0));
//    ImVec2 size = ImGui::GetItemRectSize();
//    ImVec2 position = ImGui::GetItemRectMin();
//
//    ImVec2 topLeft = position;
//    ImVec2 topRight = ImVec2(position.x + size.x, position.y);
//    ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
//    ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);
//
//    ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
//    ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);
//
//    size.x += position.x;
//    size.y += position.y;
//
//    bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
//    bool isClicked = isHovered && ImGui::IsMouseClicked(0);
//
//    if (isClicked)
//    {
//        ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
//    }
//
//    if (isHovered)
//    {
//        ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
//
//        if (ImGui::IsMouseReleased(0))
//        {
//            *releaseTriggered = true;
//        }
//    }
//
//    ImGui::SameLine();
//    ImGui::Text(slotLabel.c_str());
//
//    SameLine(GetWindowWidth() - 60);
//    if (ImGui::Button(("clear##" + slotLabel).c_str()))
//    {
//        *clearClicked = true;
//    }
//
//    return isClicked;
//}
//
//
//
//
//bool ImGui::Slot2(const std::string slotLabel, const std::string slotText, SlotData* data)
//{
//    ImVec2 windowSize = ImGui::GetWindowSize();
//    windowSize.x = std::min(std::max(windowSize.x - 100.0f, 50.0f), 250.0f);
//
//    ImGui::ButtonEx(slotText.c_str(), ImVec2(windowSize.x, 0));
//    ImVec2 size = ImGui::GetItemRectSize();
//    ImVec2 position = ImGui::GetItemRectMin();
//
//    ImVec2 topLeft = position;
//    ImVec2 topRight = ImVec2(position.x + size.x, position.y);
//    ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
//    ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);
//
//    ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
//    ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);
//
//    size.x += position.x;
//    size.y += position.y;
//
//    data->isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
//    data->isClicked = data->isHovered && ImGui::IsMouseClicked(0);
//
//    if (data->isClicked)
//    {
//        ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
//    }
//
//    if (data->isHovered)
//    {
//        ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
//
//        if (ImGui::IsMouseReleased(0))
//        {
//            data->releaseTriggered = true;
//        }
//    }
//
//    ImGui::SameLine();
//    ImGui::Text(slotLabel.c_str());
//
//    if (data->isHovered)
//    {
//        // 'c' key pressed
//        if (ImGui::IsKeyPressed(67, false))
//        {
//            data->clearClicked = true;
//        }
//    }
//
//    return data->isHovered || data->isClicked || data->releaseTriggered || data->clearClicked;
//}
//
//
//
//
//bool ImGui::ImageSlot(const std::string slotLabel, GLuint texture, bool* releaseTriggered, bool* clearClicked)
//{
//    ImGui::ImageButton((void*)(intptr_t)texture, ImVec2(80, 80), ImVec2(1, 1), ImVec2(0, 0), 0, ImVec4(1, 1, 1, 1),
//        ImVec4(1, 1, 1, 0.5));
//
//    ImVec2 size = ImGui::GetItemRectSize();
//    ImVec2 position = ImGui::GetItemRectMin();
//
//    ImVec2 topLeft = position;
//    ImVec2 topRight = ImVec2(position.x + size.x, position.y);
//    ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
//    ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);
//
//    ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
//    ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);
//
//    size.x += position.x;
//    size.y += position.y;
//
//    bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
//    bool isClicked = isHovered && ImGui::IsMouseClicked(0);
//
//    if (isClicked)
//    {
//        ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
//    }
//
//    if (isHovered)
//    {
//        ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
//
//        if (ImGui::IsMouseReleased(0))
//        {
//            *releaseTriggered = true;
//        }
//    }
//
//    ImGui::SameLine();
//    ImGui::Text(slotLabel.c_str());
//
//    SameLine(GetWindowWidth() - 60);
//    if (ImGui::Button(("clear##"+ slotLabel).c_str()))
//    {
//        *clearClicked = true;
//    }
//
//    return isClicked;
//}
//
//
//bool ImGui::ImageSlot2(const std::string slotLabel, GLuint texture, SlotData* data)
//{
//    ImGui::ImageButton((void*)(intptr_t)texture, ImVec2(80, 80), ImVec2(1, 1), ImVec2(0, 0), 0, ImVec4(1, 1, 1, 1),
//        ImVec4(1, 1, 1, 0.5));
//
//    ImVec2 size = ImGui::GetItemRectSize();
//    ImVec2 position = ImGui::GetItemRectMin();
//
//    ImVec2 topLeft = position;
//    ImVec2 topRight = ImVec2(position.x + size.x, position.y);
//    ImVec2 bottomLeft = ImVec2(position.x, position.y + size.y);
//    ImVec2 bottomRight = ImVec2(position.x + size.x, position.y + size.y);
//
//    ImGui::GetForegroundDrawList()->AddLine(topLeft, topRight, 0xFF0A0A0A);
//    ImGui::GetForegroundDrawList()->AddLine(topRight, bottomRight, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomRight, bottomLeft, 0xFF333333);
//    ImGui::GetForegroundDrawList()->AddLine(bottomLeft, topLeft, 0xFF333333);
//
//    size.x += position.x;
//    size.y += position.y;
//
//    data->isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
//    data->isClicked = data->isHovered && ImGui::IsMouseClicked(0);
//
//    if (data->isClicked)
//    {
//        ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
//    }
//
//    if (data->isHovered)
//    {
//        ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
//
//        if (ImGui::IsMouseReleased(0))
//        {
//            data->releaseTriggered = true;
//        }
//    }
//
//    ImGui::SameLine();
//    ImGui::Text(slotLabel.c_str());
//
//    if (data->isHovered)
//    {
//        // 'c' key pressed
//        if (ImGui::IsKeyPressed(67, false))
//        {
//            data->clearClicked = true;
//        }
//    }
//
//    return data->isHovered || data->isClicked || data->releaseTriggered || data->clearClicked;
//
//
//
//
//
//
//    /*bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly);
//    bool isClicked = isHovered && ImGui::IsMouseClicked(0);
//
//    if (isClicked)
//    {
//        ImGui::GetForegroundDrawList()->AddRect(position, size, 0xFFFF0000);
//    }
//
//    if (isHovered)
//    {
//        ImGui::GetForegroundDrawList()->AddRectFilled(position, size, 0x44FF0000);
//
//        if (ImGui::IsMouseReleased(0))
//        {
//            *releaseTriggered = true;
//        }
//    }
//
//    ImGui::SameLine();
//    ImGui::Text(slotLabel.c_str());*/
//
//    /*SameLine(GetWindowWidth() - 60);
//    if (ImGui::Button(("clear##" + slotLabel).c_str()))
//    {
//        *clearClicked = true;
//    }
//
//    return isClicked;*/
//}










//bool ImGui::TreeNodeExInput(const char* str_id, bool* edited, ImGuiTreeNodeFlags flags, char* buf,
//    size_t buf_size)
//{
//    ImGuiContext& g = *GImGui;
//    ImGuiWindow* window = g.CurrentWindow;
//    ImVec2 pos_before = window->DC.CursorPos;
//
//    PushID(str_id);
//    PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g.Style.ItemSpacing.x, g.Style.FramePadding.y * 2.0f));
//    bool ret = TreeNodeEx("##TreeEx", flags | ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_AllowItemOverlap);
//    PopStyleVar();
//
//    ImGuiID id = window->GetID("##Input");
//    bool temp_input_is_active = TempInputIsActive(id);
//    bool temp_input_start = ret ? IsMouseDoubleClicked(0) : false;
//
//    if (temp_input_start)
//        SetActiveID(id, window);
//
//    if (temp_input_is_active || temp_input_start)
//    {
//        ImVec2 pos_after = window->DC.CursorPos;
//        window->DC.CursorPos = pos_before;
//        /*ret = TempInputText(window->DC.LastItemRect, id, "##Input", buf, (int)buf_size, ImGuiInputTextFlags_None);*/
//        ret = TempInputText(g.LastItemData.Rect, id, "##Input", buf, (int)buf_size, ImGuiInputTextFlags_None);
//        window->DC.CursorPos = pos_after;
//
//        *edited = true;
//    }
//    else
//    {
//        window->DrawList->AddText(pos_before, GetColorU32(ImGuiCol_Text), buf);
//
//        *edited = false;
//    }
//
//    PopID();
//    return ret;
//}



bool ImGui::SelectableInput(const char *str_id, bool selected, bool *edited, ImGuiSelectableFlags flags, char *buf,
                            size_t buf_size)
{
    ImGuiContext &g = *GImGui;
    ImGuiWindow *window = g.CurrentWindow;
    ImVec2 pos_before = window->DC.CursorPos;

    PushID(str_id);
    PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g.Style.ItemSpacing.x, g.Style.FramePadding.y * 2.0f));
    bool ret = Selectable("##Selectable", selected,
                          flags | ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_AllowItemOverlap);
    PopStyleVar();

    ImGuiID id = window->GetID("##Input");
    bool temp_input_is_active = TempInputIsActive(id);
    bool temp_input_start = ret ? IsMouseDoubleClicked(0) : false;

    if (temp_input_start)
        SetActiveID(id, window);

    if (temp_input_is_active || temp_input_start)
    {
        ImVec2 pos_after = window->DC.CursorPos;
        window->DC.CursorPos = pos_before;
        /*ret = TempInputText(window->DC.LastItemRect, id, "##Input", buf, (int)buf_size, ImGuiInputTextFlags_None);*/
        ret = TempInputText(g.LastItemData.Rect, id, "##Input", buf, (int)buf_size, ImGuiInputTextFlags_None);
        window->DC.CursorPos = pos_after;

        *edited = true;
    }
    else
    {
        window->DrawList->AddText(pos_before, GetColorU32(ImGuiCol_Text), buf);

        *edited = false;
    }

    PopID();
    return ret;
}

bool ImGui::Splitter(bool split_vertically, float thickness, float *size1, float *size2, float min_size1,
                     float min_size2, float splitter_long_axis_size)
{
    ImGuiContext &g = *GImGui;
    ImGuiWindow *window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");

    ImRect bb;
    bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    bb.Max = bb.Min + CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size)
                                                    : ImVec2(splitter_long_axis_size, thickness),
                                   0.0f, 0.0f);

    return SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size1, size2, min_size1, min_size2,
                            0.0f);
}