#include "../../include/views/ProjectWindow.h"
#include "../../include/FileSystemUtil.h"
#include "../../include/EditorCameraSystem.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

ProjectWindow::ProjectWindow() : PopupWindow("##ProjectWindow", 500.0f, 200.0f, 1920.0f, 1080.0f)
{
    mMode = ProjectWindowMode::OpenProject;

    mInputBuffer.resize(256);

    mFilebrowser.setMode(FilebrowserMode::SelectFolder);
}

ProjectWindow::~ProjectWindow()
{
}

void ProjectWindow::init(Clipboard &clipboard)
{
}

void ProjectWindow::update(Clipboard &clipboard)
{
    float windowWidth = ImGui::GetWindowWidth();

    if (mMode == ProjectWindowMode::NewProject)
    {
        renderNewMode(clipboard);
    }
    else if (mMode == ProjectWindowMode::OpenProject)
    {
        renderOpenMode(clipboard);
    }
}

std::string ProjectWindow::getProjectName() const
{
    return std::string(mInputBuffer.data());
}

std::string ProjectWindow::getSelectedFolderPath() const
{
    return mFilebrowser.getSelectedFolderPath();
}

void ProjectWindow::renderNewMode(Clipboard& clipboard)
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

    bool openSelectFolderBrowser = false;
    if (ImGui::Button("Select Folder"))
    {
        openSelectFolderBrowser = true;
    }

    ImGui::SameLine();
    ImGui::Text(mFilebrowser.getSelectedFolderPath().c_str());

    mFilebrowser.render(mFilebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

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
                clipboard.setActiveProject(name, path);// openProject(name, path);
                clipboard.setActiveScene("", "", Guid::INVALID);
                /*clipboard.openScene("", "", "", "", Guid::INVALID);*/
                //clipboard.openScene("", "");
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

void ProjectWindow::renderOpenMode(Clipboard& clipboard)
{
    bool openSelectFolderBrowser = false;
    if (ImGui::Button("Select Folder"))
    {
        openSelectFolderBrowser = true;
    }

    ImGui::SameLine();
    ImGui::Text(mFilebrowser.getSelectedFolderPath().c_str());

    mFilebrowser.render(mFilebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

    // only allow the open button to be clicked if the selected folder path meets basic criteria for it being a legit
    // project folder
    bool meetsProjectCriteria = doesDirectoryExist(mFilebrowser.getSelectedFolderPath() + "\\data");

    if (!meetsProjectCriteria)
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }

    if (ImGui::Button("Open Project"))
    {
        clipboard.setActiveProject(getProjectName(), getSelectedFolderPath());
        clipboard.setActiveScene("", "", Guid::INVALID);

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
    this->mMode = mode;
}