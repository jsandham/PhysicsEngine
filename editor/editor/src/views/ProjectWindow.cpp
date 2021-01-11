#include "../../include/views/ProjectWindow.h"
#include "../../include/FileSystemUtil.h"
#include "../../include/EditorCameraSystem.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

ProjectWindow::ProjectWindow() : PopupWindow("##ProjectWindow", 500.0f, 200.0f, 1920.0f, 1080.0f)
{
    mode = ProjectWindowMode::OpenProject;

    inputBuffer.resize(256);

    filebrowser.setMode(FilebrowserMode::SelectFolder);
}

ProjectWindow::~ProjectWindow()
{
}

void ProjectWindow::init(EditorClipboard &clipboard)
{
}

void ProjectWindow::update(EditorClipboard &clipboard)
{
    float windowWidth = ImGui::GetWindowWidth();

    if (mode == ProjectWindowMode::NewProject)
    {
        renderNewMode(clipboard);
    }
    else if (mode == ProjectWindowMode::OpenProject)
    {
        renderOpenMode(clipboard);
    }
}

std::string ProjectWindow::getProjectName() const
{
    return std::string(inputBuffer.data());
}

std::string ProjectWindow::getSelectedFolderPath() const
{
    return filebrowser.getSelectedFolderPath();
}

void ProjectWindow::renderNewMode(EditorClipboard& clipboard)
{
    float projectNameTitleWidth = 100.0f;
    float inputTextWidth = 400.0f;

    ImGui::SetNextItemWidth(projectNameTitleWidth);
    ImGui::Text("Project Name");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputTextWidth);
    if (ImGui::InputText("##Project Name", &inputBuffer[0], (int)inputBuffer.size(),
                         ImGuiInputTextFlags_EnterReturnsTrue))
    {
    }

    bool openSelectFolderBrowser = false;
    if (ImGui::Button("Select Folder"))
    {
        openSelectFolderBrowser = true;
    }

    ImGui::SameLine();
    ImGui::Text(filebrowser.getSelectedFolderPath().c_str());

    filebrowser.render(filebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

    if (ImGui::Button("Create Project"))
    {
        std::string name = getProjectName();
        std::string path = getSelectedFolderPath() + "\\" + getProjectName();

        if (PhysicsEditor::createDirectory(path))
        {
            bool success = true;
            success &= createDirectory(path + "\\data");
            success &= createDirectory(path + "\\data\\scenes");
            success &= createDirectory(path + "\\data\\textures");
            success &= createDirectory(path + "\\data\\meshes");
            success &= createDirectory(path + "\\data\\materials");
            success &= createDirectory(path + "\\data\\shaders");

            if (success)
            {
                clipboard.openProject(name, path);
                clipboard.openScene("", "", "", "", Guid::INVALID);
            }
            else
            {
                Log::error("Could not create project sub directories\n");
                return;
            }
        }
        else
        {
            Log::error("Could not create project root directory\n");
            return;
        }

        // mark any (non-editor) entities in currently opened scene to be latent destroyed
        clipboard.getWorld()->latentDestroyEntitiesInWorld();

        // tell library directory which project to watch
        clipboard.getLibrary().watch(path);

        // reset editor camera
        clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

        ImGui::CloseCurrentPopup();
    }
}

void ProjectWindow::renderOpenMode(EditorClipboard& clipboard)
{
    bool openSelectFolderBrowser = false;
    if (ImGui::Button("Select Folder"))
    {
        openSelectFolderBrowser = true;
    }

    ImGui::SameLine();
    ImGui::Text(filebrowser.getSelectedFolderPath().c_str());

    filebrowser.render(filebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

    // only allow the open button to be clicked if the selected folder path meets basic criteria for it being a legit
    // project folder
    bool meetsProjectCriteria = doesDirectoryExist(filebrowser.getSelectedFolderPath() + "\\data");

    if (!meetsProjectCriteria)
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }

    if (ImGui::Button("Open Project"))
    {
        clipboard.openProject(getProjectName(), getSelectedFolderPath());
        clipboard.openScene("", "", "", "", Guid::INVALID);

        // mark any (non-editor) entities in currently opened scene to be latent destroyed
        clipboard.getWorld()->latentDestroyEntitiesInWorld();

        // tell library directory which project to watch
        clipboard.getLibrary().watch(getSelectedFolderPath());

        // reset editor camera
        clipboard.getWorld()->getSystem<EditorCameraSystem>()->resetCamera();

        ImGui::CloseCurrentPopup();
    }

    if (!meetsProjectCriteria)
    {
        ImGui::PopItemFlag();
        ImGui::PopStyleVar();
    }
}

void ProjectWindow::setMode(ProjectWindowMode mode)
{
    this->mode = mode;
}