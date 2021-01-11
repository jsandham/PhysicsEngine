#include "../../include/views/AboutPopup.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

AboutPopup::AboutPopup() : PopupWindow("About", 0.0f, 0.0f, 500.0f, 500.0f)
{
}

AboutPopup::~AboutPopup()
{
}

void AboutPopup::init(EditorClipboard &clipboard)
{
}

void AboutPopup::update(EditorClipboard &clipboard)
{
    ImGui::Text("About PhysicsEngine");
    ImGui::TextWrapped("About engine text goes here");

    if (ImGui::Button("Ok"))
    {
        ImGui::CloseCurrentPopup();
    }
}