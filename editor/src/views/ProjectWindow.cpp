#include "../../include/views/ProjectWindow.h"
#include "../../include/ProjectDatabase.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

extern std::string selectFolder();

ProjectWindow::ProjectWindow()
{
    mInputBuffer.resize(256);
}

ProjectWindow::~ProjectWindow()
{
}

void ProjectWindow::init(Clipboard &clipboard)
{
}

void ProjectWindow::update(Clipboard &clipboard, bool isOpenedThisFrame)
{
    if (isOpenedThisFrame)
    {
        ImGui::SetNextWindowPos(ImVec2(600.0f, 300.0f));
        ImGui::SetNextWindowSize(ImVec2(500.0f, 200.0f));

        ImGui::OpenPopup("##ProjectWindow");
        mOpen = true;
    }

    if (ImGui::BeginPopupModal("##ProjectWindow", &mOpen, ImGuiWindowFlags_NoResize))
    {
        float projectNameTitleWidth = 100.0f;
        float inputTextWidth = 400.0f;

        ImGui::SetNextItemWidth(projectNameTitleWidth);
        ImGui::Text("Project Name");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(inputTextWidth);
        if (ImGui::InputText("##Project Name", &mInputBuffer[0], (int)mInputBuffer.size(),
            ImGuiInputTextFlags_EnterReturnsTrue))
        {
        }

        if (ImGui::Button("Select Folder"))
        {
            mSelectedFolder = selectFolder();
        }

        ImGui::SameLine();
        ImGui::Text(mSelectedFolder.c_str());

        if (ImGui::Button("Create Project"))
        {
            ProjectDatabase::newProject(clipboard, std::filesystem::path(mSelectedFolder) / getProjectName());

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

std::string ProjectWindow::getProjectName() const
{
    return std::string(mInputBuffer.data());
}