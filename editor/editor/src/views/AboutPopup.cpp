#include "../../include/views/AboutPopup.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

AboutPopup::AboutPopup()
{
 
}

AboutPopup::~AboutPopup()
{
}

void AboutPopup::init(EditorClipboard& clipboard)
{
}

void AboutPopup::update(EditorClipboard& clipboard, bool isOpenedThisFrame)
{
    this->Window::update(clipboard, isOpenedThisFrame);

    if (!windowActive)
    {
        return;
    }

    if (isOpenedThisFrame)
    {
        ImGui::SetNextWindowSize(ImVec2(500, 500));
        ImGui::OpenPopup("About");
    }

    if (ImGui::BeginPopupModal("About"))
    {
        ImGui::Text("About PhysicsEngine");
        ImGui::TextWrapped("About engine text goes here");

        if (ImGui::Button("Ok"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}