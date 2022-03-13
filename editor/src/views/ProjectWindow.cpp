#include "../../include/views/ProjectWindow.h"
#include "../../include/EditorProjectManager.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

extern std::string selectFolder();

ProjectWindow::ProjectWindow() : PopupWindow("##ProjectWindow", 600.0f, 300.0f, 500.0f, 200.0f)
{
    mInputBuffer.resize(256);
}

ProjectWindow::~ProjectWindow()
{
}

void ProjectWindow::init(Clipboard &clipboard)
{
}

void ProjectWindow::update(Clipboard &clipboard)
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
        EditorProjectManager::newProject(clipboard, std::filesystem::path(mSelectedFolder) / getProjectName());

        ImGui::CloseCurrentPopup();
    }
}

std::string ProjectWindow::getProjectName() const
{
    return std::string(mInputBuffer.data());
}