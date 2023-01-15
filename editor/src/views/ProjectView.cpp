#include "../../include/views/ProjectView.h"
#include "../../include/EditorSceneManager.h"
#include "../../include/ProjectDatabase.h"

#include <algorithm>
#include <stack>
#include <fstream>
#include <assert.h>

#include "../../include/imgui/imgui_extensions.h"

#include "core/Guid.h"
#include "core/Log.h"

using namespace PhysicsEditor;

ProjectView::ProjectView() : Window("Project View")
{
    mHighlightedType = InteractionType::None;
    mHighlightedPath = std::filesystem::path();
    mHoveredPath = std::filesystem::path();

    mSelectedDirectoryPath = std::filesystem::path();
    mSelectedFilePath = std::filesystem::path();
}

ProjectView::~ProjectView()
{

}

void ProjectView::init(Clipboard &clipboard)
{
}

void ProjectView::update(Clipboard &clipboard)
{
    clipboard.mOpen[static_cast<int>(View::ProjectView)] = isOpen();
    clipboard.mHovered[static_cast<int>(View::ProjectView)] = isHovered();
    clipboard.mFocused[static_cast<int>(View::ProjectView)] = isFocused();
    clipboard.mOpenedThisFrame[static_cast<int>(View::ProjectView)] = openedThisFrame();
    clipboard.mHoveredThisFrame[static_cast<int>(View::ProjectView)] = hoveredThisFrame();
    clipboard.mFocusedThisFrame[static_cast<int>(View::ProjectView)] = focusedThisFrame();
    clipboard.mClosedThisFrame[static_cast<int>(View::ProjectView)] = closedThisFrame();
    clipboard.mUnfocusedThisFrame[static_cast<int>(View::ProjectView)] = unfocusedThisFrame();
    clipboard.mUnhoveredThisFrame[static_cast<int>(View::ProjectView)] = unhoveredThisFrame();

    ImGui::Text(("mHighlightedType: " + std::to_string(static_cast<int>(mHighlightedType))).c_str());
    ImGui::Text(("mHighlightedPath: " + mHighlightedPath.string()).c_str());
    ImGui::Text(("mHoveredPath: " + mHoveredPath.string()).c_str());
    ImGui::Text(("mSelectedDirectoryPath: " + mSelectedDirectoryPath.string()).c_str());
    ImGui::Text(("mSelectedFilePath: " + mSelectedFilePath.string()).c_str());

    if (!clipboard.getProjectPath().empty())
    {
        mProjectTree.buildProjectTree(clipboard.getProjectPath());

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
        std::stack<ProjectNode*> stack;

        stack.push(mProjectTree.getRoot());

        while (!stack.empty())
        {
            ProjectNode* current = stack.top();
            stack.pop();

            std::error_code errorCode;
            if (std::filesystem::equivalent(mSelectedDirectoryPath, current->getDirectoryPath(), errorCode))
            {
                directories = current->getChildren();
                fileLabels = current->getFileLabels();
                filePaths = current->getFilePaths();
                fileTypes = current->getFileTypes();

                break;
            }
            
            for (size_t j = 0; j < current->getChildCount(); j++)
            {
                stack.push(current->getChild(j));
            }
        }
    }

    // draw directories in right pane
    for (size_t i = 0; i < directories.size(); i++)
    {
        std::error_code errorCode;
        if (ImGui::Selectable(directories[i]->getDirectoryLabel().c_str(), std::filesystem::equivalent(directories[i]->getDirectoryPath(), mHighlightedPath, errorCode), ImGuiSelectableFlags_AllowDoubleClick))
        {
            mHighlightedType = InteractionType::Folder;
            mHighlightedPath = directories[i]->getDirectoryPath();

            if (ImGui::IsMouseDoubleClicked(0))
            {
                mSelectedDirectoryPath = directories[i]->getDirectoryPath();

                mFilter.Clear();
            }
        }

        if (ImGui::IsItemHovered())
        {
            mHoveredPath = directories[i]->getDirectoryPath();
        }

        if (ImGui::BeginDragDropSource())
        {
            std::string directoryPath = directories[i]->getDirectoryPath().string();

            const void* data = static_cast<const void*>(directoryPath.c_str());

            ImGui::SetDragDropPayload("FOLDER", data, directoryPath.length() + 1);
            ImGui::Text(directoryPath.c_str());
            ImGui::EndDragDropSource();
        }

        if (ImGui::BeginDragDropTarget())
        {
            const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("FOLDER");
            if (payload != nullptr)
            {
                const char* data = static_cast<const char*>(payload->Data);

                std::filesystem::path incomingPath = std::string(data);
                std::filesystem::path currentPath = directories[i]->getDirectoryPath();
                std::filesystem::path newPath = currentPath / incomingPath.filename();

                ProjectDatabase::move(incomingPath, newPath);
            }

            /*payload = ImGui::AcceptDragDropPayload("MATERIAL");
            if (payload != nullptr)
            {
                   
            }*/
            
            ImGui::EndDragDropTarget();
        }
    }

    // draw files in right pane
    for (size_t i = 0; i < filePaths.size(); i++)
    {
        std::error_code errorCode;
        if (ImGui::Selectable(fileLabels[i].c_str(), std::filesystem::equivalent(filePaths[i], mHighlightedPath, errorCode), ImGuiSelectableFlags_AllowDoubleClick))
        {
            mHighlightedType = InteractionType::File;
            mHighlightedPath = filePaths[i];

            if (ImGui::IsMouseDoubleClicked(0))
            {
                if (fileTypes[i] == InteractionType::Scene)
                {
                    EditorSceneManager::openScene(clipboard, filePaths[i]);
                }

                mSelectedFilePath = filePaths[i];

                clipboard.mSelectedType = fileTypes[i];
                clipboard.mSelectedPath = filePaths[i];
                clipboard.mSelectedId = ProjectDatabase::getGuid(clipboard.mSelectedPath);

                mFilter.Clear();
            }
        }

        if (ImGui::IsItemHovered())
        {
            mHoveredPath = filePaths[i];
        }

        if (ImGui::BeginDragDropSource())
        {
            std::string filePath = filePaths[i].string();
            const void* data = static_cast<const void*>(filePath.c_str());
            switch (fileTypes[i])
            {
            case InteractionType::Cubemap:
                ImGui::SetDragDropPayload("CUBEMAP_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::Texture2D:
                ImGui::SetDragDropPayload("TEXTURE2D_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::Mesh:
                ImGui::SetDragDropPayload("MESH_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::Material:
                ImGui::SetDragDropPayload("MATERIAL_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::Scene:
                ImGui::SetDragDropPayload("SCENE_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::Shader:
                ImGui::SetDragDropPayload("SHADER_PATH", data, filePath.length() + 1);
                break;
            case InteractionType::File:
                ImGui::SetDragDropPayload("FILE_PATH", data, filePath.length() + 1);
                break;
            }
            ImGui::Text(filePath.c_str());
            ImGui::EndDragDropSource();
        }
    }

    if (!mSelectedDirectoryPath.empty())
    {
        // Right click popup menu
        if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
        {
            if (ImGui::BeginMenu("Create..."))
            {
                if (ImGui::MenuItem("Folder"))
                {
                    ProjectDatabase::createDirectory(mSelectedDirectoryPath);
                }

                ImGui::Separator();

                if (ImGui::BeginMenu("Shader..."))
                {
                    if (ImGui::MenuItem("GLSL"))
                    {
                        ProjectDatabase::createShaderFile(mSelectedDirectoryPath);
                    }

                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem("Cubemap"))
                {
                    ProjectDatabase::createCubemapFile(clipboard.getWorld(), mSelectedDirectoryPath);
                }

                if (ImGui::MenuItem("Material"))
                {
                    ProjectDatabase::createMaterialFile(clipboard.getWorld(), mSelectedDirectoryPath);
                }

                if (ImGui::MenuItem("Sprite"))
                {
                    ProjectDatabase::createSpriteFile(clipboard.getWorld(), mSelectedDirectoryPath);
                }

                if (ImGui::MenuItem("RenderTexture"))
                {
                    ProjectDatabase::createRenderTextureFile(clipboard.getWorld(), mSelectedDirectoryPath);
                }

                ImGui::EndMenu();
            }

            std::error_code errorCode;
            bool showDelete = !mHighlightedPath.empty() && std::filesystem::equivalent(mHighlightedPath, mHoveredPath, errorCode);
            if (ImGui::MenuItem("Delete", nullptr, false, showDelete))
            {
                if (mHighlightedType == InteractionType::Folder)
                {
                    if (std::filesystem::remove_all(mHighlightedPath))
                    {
                        //clipboard.mSelectedType = InteractionType::None;
                        //clipboard.mSelectedId = PhysicsEngine::Guid::INVALID;
                        //clipboard.mSelectedPath = std::filesystem::path();

                        //clipboard.clearSelectedItem();

                        //if (mHighlightedPath == mSelectedDirectoryPath)
                        //{
                        //    mSelectedDirectoryPath = std::filesystem::path();
                        //}
                    }
                }
                else
                {
                    if (std::filesystem::equivalent(clipboard.mSelectedPath, mHighlightedPath))
                    {
                        clipboard.mSelectedType = InteractionType::None;
                        clipboard.mSelectedId = PhysicsEngine::Guid::INVALID;
                        clipboard.mSelectedPath = std::filesystem::path();

                        if (std::filesystem::remove(mHighlightedPath))
                        {
                            //clipboard.mSelectedType = InteractionType::None;
                            //clipboard.mSelectedId = PhysicsEngine::Guid::INVALID;
                            //clipboard.mSelectedPath = std::filesystem::path();

                            ////clipboard.clearSelectedItem();

                            //if (mHighlightedPath == mSelectedFilePath)
                            //{
                            //    mSelectedFilePath = std::filesystem::path();
                            //}
                        }
                    }
                }
            }

            ImGui::EndPopup();
        }
    }
}

void ProjectView::drawProjectTree()
{
    drawProjectNodeRecursive(mProjectTree.getRoot());
}

void ProjectView::drawProjectNodeRecursive(ProjectNode *node)
{
    if (node != nullptr)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;
        if (node->getChildCount() == 0)
        {
            node_flags |= ImGuiTreeNodeFlags_Leaf;
        }

        std::error_code errorCode;
        if (std::filesystem::equivalent(mSelectedDirectoryPath, node->getDirectoryPath(), errorCode))
        {
            node_flags |= ImGuiTreeNodeFlags_Selected;
        }

        bool open = ImGui::TreeNodeEx(node->getDirectoryLabel().c_str(), node_flags);

        if (ImGui::BeginDragDropSource())
        {
            std::string directoryPath = node->getDirectoryPath().string();

            const void* data = static_cast<const void*>(directoryPath.c_str());

            ImGui::SetDragDropPayload("FOLDER", data, directoryPath.length() + 1);
            ImGui::Text(directoryPath.c_str());
            ImGui::EndDragDropSource();
        }

        if (ImGui::BeginDragDropTarget())
        {
            const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("FOLDER");
            if (payload != nullptr)
            {
                const char* data = static_cast<const char*>(payload->Data);

                std::filesystem::path incomingPath = std::string(data);
                std::filesystem::path currentPath = node->getDirectoryPath();
                std::filesystem::path newPath = currentPath / incomingPath.filename();

                ProjectDatabase::move(incomingPath, newPath);
            }
            ImGui::EndDragDropTarget();
        }

        if (ImGui::IsItemHovered())
        {
            if (ImGui::IsMouseReleased(ImGuiMouseButton_::ImGuiMouseButton_Left))
            {
                mHighlightedType = InteractionType::Folder;
                mHighlightedPath = node->getDirectoryPath();
                mSelectedDirectoryPath = node->getDirectoryPath();
                mFilter.Clear();
            }
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
}