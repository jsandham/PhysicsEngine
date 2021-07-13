#include "../../include/views/ProjectWindow.h"
#include "../../include/EditorCameraSystem.h"

#include "imgui.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

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

std::filesystem::path ProjectWindow::getSelectedFolderPath() const
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
    ImGui::Text(mFilebrowser.getSelectedFolderPath().string().c_str());

    mFilebrowser.render(mFilebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

    if (ImGui::Button("Create Project"))
    {
        std::filesystem::path path = getSelectedFolderPath() / getProjectName();

        if (std::filesystem::create_directory(path))
        {
            bool success = true;
            success &= std::filesystem::create_directory(path / "data");
            success &= std::filesystem::create_directory(path / "data/scenes");
            success &= std::filesystem::create_directory(path / "data/textures");
            success &= std::filesystem::create_directory(path / "data/meshes");
            success &= std::filesystem::create_directory(path / "data/materials");
            success &= std::filesystem::create_directory(path / "data/shaders");
            success &= std::filesystem::create_directory(path / "data/sprites");

            if (success)
            {
                clipboard.setActiveProject(getProjectName(), path.string());
                clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);
            }
            else
            {
                PhysicsEngine::Log::error("Could not create project sub directories\n");
                return;
            }
        }
        else
        {
            PhysicsEngine::Log::error("Could not create project root directory\n");
            return;
        }

        // mark any (non-editor) entities in currently opened scene to be latent destroyed
        clipboard.getWorld()->latentDestroyEntitiesInWorld();

        // tell library directory which project to watch
        clipboard.getLibrary().watch(path.string());

        // reset editor camera
        clipboard.getWorld()->getSystem<PhysicsEngine::EditorCameraSystem>()->resetCamera();

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
    ImGui::Text(mFilebrowser.getSelectedFolderPath().string().c_str());

    mFilebrowser.render(mFilebrowser.getSelectedFolderPath(), openSelectFolderBrowser);

    // only allow the open button to be clicked if the selected folder path meets basic criteria for it being a legit
    // project folder
    bool meetsProjectCriteria = std::filesystem::exists(mFilebrowser.getSelectedFolderPath() / "data");

    if (!meetsProjectCriteria)
    {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }

    if (ImGui::Button("Open Project"))
    {
        clipboard.setActiveProject(getProjectName(), getSelectedFolderPath().string());
        clipboard.setActiveScene("", "", PhysicsEngine::Guid::INVALID);

        // mark any (non-editor) entities in currently opened scene to be latent destroyed
        clipboard.getWorld()->latentDestroyEntitiesInWorld();

        // tell library directory which project to watch
        clipboard.getLibrary().watch(getSelectedFolderPath().string());

        // reset editor camera
        clipboard.getWorld()->getSystem<PhysicsEngine::EditorCameraSystem>()->resetCamera();

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
    mMode = mode;
}