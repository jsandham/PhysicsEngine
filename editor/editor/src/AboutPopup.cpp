#include "../include/AboutPopup.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

AboutPopup::AboutPopup()
{
    isVisible = false;
}

AboutPopup::~AboutPopup()
{
}

void AboutPopup::render(bool becomeVisibleThisFrame)
{
    if (isVisible != becomeVisibleThisFrame)
    {
        isVisible = becomeVisibleThisFrame;

        if (isVisible)
        {
            ImGui::SetNextWindowSize(ImVec2(500, 500));
            ImGui::OpenPopup("About");
        }
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