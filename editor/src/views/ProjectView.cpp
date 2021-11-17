#include "../../include/views/ProjectView.h"
#include "../../include/EditorSceneManager.h"

#include <algorithm>
#include <stack>

#include "../../include/imgui/imgui_extensions.h"

#include "core/Guid.h"
#include "core/Log.h"

using namespace PhysicsEditor;

ProjectView::ProjectView() : Window("Project View")
{
    mSelected = nullptr;
    mRightPanelSelectedPath = std::filesystem::path();
}

ProjectView::~ProjectView()
{

}

void ProjectView::init(Clipboard &clipboard)
{
}

void ProjectView::update(Clipboard &clipboard)
{
    if (!clipboard.getProjectPath().empty())
    {
        if (!mProjectTree.isEmpty() && mProjectTree.getRoot()->getDirectoryPath() != (clipboard.getProjectPath() / "data") || mProjectTree.isEmpty())
        {
            mProjectTree.buildProjectTree(clipboard.getProjectPath());
        }
    }

    if (!clipboard.getProjectPath().empty())
    {
        mFilter.Draw("Filter", -100.0f);

        ImVec2 WindowSize = ImGui::GetWindowSize();

        static float ratio = 0.5f;

        float sz1 = ratio * WindowSize.x;
        float sz2 = (1.0f - ratio) * WindowSize.x;

        ImGui::Splitter(true, 8.0f, &sz1, &sz2, 8, 8, WindowSize.y);

        ratio = sz1 / WindowSize.x;

        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;

        if (ImGui::BeginChild("LeftPane", ImVec2(sz1, WindowSize.y), true, flags))
        {
            drawLeftPane();
        }
        ImGui::EndChild();

        ImGui::SameLine();

        if (ImGui::BeginChild("RightPane", ImVec2(sz2, WindowSize.y), true, flags))
        {
            drawRightPane(clipboard);
        }
        ImGui::EndChild();
    }
}

void ProjectView::drawLeftPane()
{
    drawProjectTree();
}

void ProjectView::drawRightPane(Clipboard &clipboard)
{
    std::vector<ProjectNode *> directories;

    std::vector<std::string> fileLabels;
    std::vector<std::filesystem::path> filePaths;
    std::vector<InteractionType> fileTypes;

    // Determine directories and files to be drawn in right pane
    if (mFilter.IsActive())
    {
        std::stack<ProjectNode*> stack;

        stack.push(mProjectTree.getRoot());

        while (!stack.empty())
        {
            ProjectNode* current = stack.top();
            stack.pop();

            if (mFilter.PassFilter(current->getDirectoryPath().filename().string().c_str()))
            {
                directories.push_back(current);
            }

            for (size_t j = 0; j < current->getFileCount(); j++)
            {
                if (mFilter.PassFilter(current->getFilePath(j).filename().string().c_str()))
                {
                    fileLabels.push_back(current->getFileLabel(j));
                    filePaths.push_back(current->getFilePath(j));
                    fileTypes.push_back(current->getFileType(j));
                }
            }

            for (size_t j = 0; j < current->getChildCount(); j++)
            {
                stack.push(current->getChild(j));
            }
        }
    }
    else
    {
        if (mSelected != nullptr)
        {
            directories = mSelected->getChildren();
           
            fileLabels = mSelected->getFileLabels();
            filePaths = mSelected->getFilePaths();
            fileTypes = mSelected->getFileTypes();
        }
    }

    ProjectNode *newSelection = nullptr;

    // draw directories in right pane
    for (size_t i = 0; i < directories.size(); i++)
    {
        if (ImGui::Selectable(directories[i]->getDirectoryLabel().c_str(), directories[i]->getDirectoryPath() == mRightPanelSelectedPath, ImGuiSelectableFlags_AllowDoubleClick))
        {
            mRightPanelSelectedPath = directories[i]->getDirectoryPath();

            if (ImGui::IsMouseDoubleClicked(0))
            {
                newSelection = directories[i];
                mFilter.Clear();
            }
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0))
        {
            clipboard.mSelectedType = InteractionType::Folder;
            clipboard.mSelectedPath = directories[i]->getDirectoryPath().string();
        }
    }

    // draw files in right pane
    for (size_t i = 0; i < filePaths.size(); i++)
    {
        if (ImGui::Selectable(fileLabels[i].c_str(), filePaths[i] == mRightPanelSelectedPath, ImGuiSelectableFlags_AllowDoubleClick))
        {
            mRightPanelSelectedPath = filePaths[i];

            if (ImGui::IsMouseDoubleClicked(0))
            {
                if (filePaths[i].extension().string() == ".scene")
                {
                    EditorSceneManager::openScene(clipboard, filePaths[i]);
                }
            }
        }

        if (ImGui::IsItemHovered())
        {
            if (ImGui::IsMouseClicked(0))
            {
                clipboard.mDraggedType = fileTypes[i];
                clipboard.mDraggedPath = filePaths[i].string();
                clipboard.mDraggedId = clipboard.mLibrary.getId(clipboard.mDraggedPath);
            }

            if (ImGui::IsMouseReleased(0))
            {
                clipboard.mSelectedType = fileTypes[i];
                clipboard.mSelectedPath = filePaths[i].string();
                clipboard.mSelectedId = clipboard.mLibrary.getId(clipboard.mSelectedPath);
            }
        }

        if (!ImGui::IsMouseDown(0))
        {
            clipboard.clearDraggedItem();
        }
    }

    if (newSelection != nullptr)
    {
        mSelected = newSelection;
    }

    if (mSelected == nullptr)
    {
        return;
    }

    // Right click popup menu
    if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
    {
        if (ImGui::BeginMenu("Create..."))
        {
            if (ImGui::MenuItem("Folder"))
            {
                size_t count = mSelected->getChildCount();
                std::string foldername = "Folder" + (count > 0 ? "(" + std::to_string(count) + ")" : "");
                std::filesystem::path folderPath = mSelected->getDirectoryPath() / foldername;
                if (std::filesystem::create_directory(folderPath))
                {
                    mSelected->addDirectory(foldername);
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Material"))
            {
                size_t count = clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>();
                std::string filename = ("NewMaterial(" + std::to_string(count) + ").material");
                std::filesystem::path filepath = mSelected->getDirectoryPath() / filename;

                PhysicsEngine::Material* material = clipboard.getWorld()->createAsset<PhysicsEngine::Material>();
                material->writeToYAML(filepath.string());

                mSelected->addFile(filename);
            }

            if (ImGui::MenuItem("Sprite"))
            {
                size_t count = clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Sprite>();
                std::string filename = ("NewSprite(" + std::to_string(count) + ").sprite");
                std::filesystem::path filepath = mSelected->getDirectoryPath() / filename;

                PhysicsEngine::Sprite* sprite = clipboard.getWorld()->createAsset<PhysicsEngine::Sprite>();
                sprite->writeToYAML(filepath.string());

                mSelected->addFile(filename);
            }

            if (ImGui::MenuItem("RenderTexture"))
            {
                size_t count = clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::RenderTexture>();
                std::string filename = ("NewRenderTexture(" + std::to_string(count) + ").rendertexture");
                std::filesystem::path filepath = mSelected->getDirectoryPath() / filename;

                PhysicsEngine::RenderTexture* renderTexture = clipboard.getWorld()->createAsset<PhysicsEngine::RenderTexture>();
                renderTexture->writeToYAML(filepath.string());

                mSelected->addFile(filename);
            }

            ImGui::EndMenu();
        }
        
        if (ImGui::MenuItem("Delete", nullptr, false, !clipboard.getSelectedPath().empty()))
        {
            if (clipboard.mSelectedType == InteractionType::Folder)
            {
                std::filesystem::path folderpath = clipboard.getSelectedPath();
                if (std::filesystem::remove_all(folderpath))
                {
                    clipboard.clearSelectedItem();

                    mSelected->removeDirectory(folderpath.filename().string());

                    if (folderpath == mRightPanelSelectedPath)
                    {
                        mRightPanelSelectedPath = std::filesystem::path();
                    }
                }
            }
            else
            {
                std::filesystem::path filepath = clipboard.getSelectedPath();
                if (std::filesystem::remove(filepath))
                {
                    clipboard.clearSelectedItem();

                    mSelected->removeFile(filepath.filename().string());
                    
                    if (filepath == mRightPanelSelectedPath)
                    {
                        mRightPanelSelectedPath = std::filesystem::path();
                    }
                }
            }
        }

        if (ImGui::MenuItem("Refresh", nullptr, false, true))
        {
            mSelected->rebuild();
        }
      
        ImGui::EndPopup();
    }
}

void ProjectView::drawProjectTree()
{
    drawProjectNodeRecursive(mProjectTree.getRoot());
}

void ProjectView::drawProjectNodeRecursive(ProjectNode *node)
{
    if (node == nullptr)
    {
        return;
    }

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;
    if (node->getChildCount() == 0)
    {
        node_flags |= ImGuiTreeNodeFlags_Leaf;
    }

    if (mSelected == node)
    {
        node_flags |= ImGuiTreeNodeFlags_Selected;
    }

    bool open = ImGui::TreeNodeEx(node->getDirectoryLabel().c_str(), node_flags);

    if (ImGui::IsItemClicked())
    {
        mSelected = node;
        mFilter.Clear();
    }

    if (open)
    {
        // recurse for each sub directory
        for (size_t i = 0; i < node->getChildCount(); i++)
        {
            drawProjectNodeRecursive(node->getChild(i));
        }

        ImGui::TreePop();
    }
}