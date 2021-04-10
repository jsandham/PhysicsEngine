#include "../../include/views/ProjectView.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

#include "../../include/imgui/imgui_extensions.h"

#include "../../include/FileSystemUtil.h"

#include <algorithm>
#include <stack>

#include "core/Guid.h"
#include "core/Log.h"

#include "../../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

ProjectView::ProjectView() : Window("Project View")
{
    root = nullptr;
    selected = nullptr;
}

ProjectView::~ProjectView()
{
    deleteProjectTree();
}

void ProjectView::init(Clipboard &clipboard)
{
}

void ProjectView::update(Clipboard &clipboard)
{
    if (!clipboard.getProjectPath().empty())
    {
        if (root != nullptr && root->directoryPath != (clipboard.getProjectPath() + "\\data") || root == nullptr /*||
            editorBecameActiveThisFrame*/)
        {
            buildProjectTree(clipboard.getProjectPath());
        }
    }

    if (!clipboard.getProjectPath().empty())
    {
        filter.Draw("Filter", -100.0f);

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
    std::vector<std::string> files;

    // Determine directories and files to be drawn in right pane
    if (filter.IsActive())
    {
        for (size_t i = 0; i < nodes.size(); i++)
        {
            for (size_t j = 0; j < nodes[i]->children.size(); j++)
            {
                if (filter.PassFilter(nodes[i]->children[j]->directoryName.c_str()))
                {
                    directories.push_back(nodes[i]->children[j]);
                }
            }

            for (size_t j = 0; j < nodes[i]->filePaths.size(); j++)
            {
                if (filter.PassFilter(nodes[i]->filePaths[j].c_str()))
                {
                    files.push_back(nodes[i]->filePaths[j]);
                }
            }
        }
    }
    else
    {
        if (selected != nullptr)
        {
            directories = selected->children;
            files = selected->filePaths;
        }
    }

    ProjectNode *newSelection = nullptr;

    // draw directories in right pane
    for (size_t i = 0; i < directories.size(); i++)
    {
        std::string icon = std::string(ICON_FA_FOLDER);
        std::string label = icon + " " + directories[i]->directoryName;

        if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick))
        {
            if (ImGui::IsMouseDoubleClicked(0))
            {
                newSelection = directories[i];
                filter.Clear();
            }
        }
    }

    // draw files in right pane
    for (size_t i = 0; i < files.size(); i++)
    {
        std::string fileName = getFileName(files[i]);
        std::string extension = getFileExtension(files[i]);
        std::string icon = std::string(ICON_FA_FILE);
        std::string label = icon + " " + fileName;

        ImGui::Selectable(label.c_str());

        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
        {
            clipboard.setDraggedItem(getInteractionTypeFromFileExtension(extension),
                                     clipboard.getLibrary().getId(files[i]));
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0))
        {
            clipboard.setSelectedItem(getInteractionTypeFromFileExtension(extension),
                                      clipboard.getLibrary().getId(files[i]));
        }

        if (!ImGui::IsMouseDown(0))
        {
            clipboard.clearDraggedItem();
        }
    }

    if (newSelection != nullptr)
    {
        selected = newSelection;
    }

    // Right click popup menu
    if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
    {
        if (ImGui::BeginMenu("Create..."))
        {
            if (ImGui::MenuItem("Material"))
            {
                size_t count = clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>();
                std::string filepath = selected->directoryPath + "\\NewMaterial" + "(" + std::to_string(count) + ")" + ".material";

                PhysicsEngine::Material* material = clipboard.getWorld()->createAsset<PhysicsEngine::Material>();
                material->writeToYAML(filepath);

                selected->filePaths.push_back(filepath);
            }

            ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Delete", nullptr, false, clipboard.getSelectedId().isValid()))
        {
            std::string filepath = clipboard.getLibrary().getFile(clipboard.getSelectedId());
            if (PhysicsEditor::deleteFile(filepath))
            {
                clipboard.clearSelectedItem();

                for (size_t i = 0; i < selected->filePaths.size(); i++)
                {
                    if (selected->filePaths[i] == filepath) {
                        selected->filePaths.erase(selected->filePaths.begin() + i);
                        break;
                    }
                }
            }
        }

        ImGui::EndPopup();
    }
}

void ProjectView::deleteProjectTree()
{
    if (root == nullptr)
    {
        return;
    }

    selected = nullptr;

    nodes.clear();

    std::stack<ProjectNode *> stack;
    stack.push(root);
    while (!stack.empty())
    {
        ProjectNode *current = stack.top();
        stack.pop();

        for (size_t i = 0; i < current->children.size(); i++)
        {
            stack.push(current->children[i]);
        }

        delete current;
    }
}

void ProjectView::buildProjectTree(const std::string &currentProjectPath)
{
    deleteProjectTree();

    int id = -1;

    root = new ProjectNode();
    root->id = ++id;
    root->parent = nullptr;
    root->directoryName = "data";
    root->directoryPath = currentProjectPath + "\\data";
    root->filePaths = getFilesInDirectory(root->directoryPath, true);

    nodes.push_back(root);

    std::stack<ProjectNode *> stack;
    stack.push(root);

    while (!stack.empty())
    {
        ProjectNode *current = stack.top();
        stack.pop();

        // find directories that exist in the current directory
        std::vector<std::string> subDirectoryPaths =
            PhysicsEditor::getDirectoriesInDirectory(current->directoryPath, true);
        std::vector<std::string> subDirectoryNames =
            PhysicsEditor::getDirectoriesInDirectory(current->directoryPath, false);

        // recurse for each sub directory
        for (size_t i = 0; i < subDirectoryPaths.size(); i++)
        {
            ProjectNode *child = new ProjectNode();
            child->id = ++id;
            child->parent = current;
            child->directoryName = subDirectoryNames[i];
            child->directoryPath = subDirectoryPaths[i];
            child->filePaths = PhysicsEditor::getFilesInDirectory(child->directoryPath, true);

            current->children.push_back(child);

            stack.push(child);
            nodes.push_back(child);
        }
    }
}

void ProjectView::drawProjectTree()
{
    drawProjectNodeRecursive(root);
}

void ProjectView::drawProjectNodeRecursive(ProjectNode *node)
{
    if (node == nullptr)
    {
        return;
    }

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
    std::string icon = std::string(ICON_FA_FOLDER);
    if (node->children.empty())
    {
        if (node->filePaths.empty())
        {
            icon = std::string(ICON_FA_FOLDER_O);
        }

        node_flags |= ImGuiTreeNodeFlags_Leaf;
    }

    std::string label = icon + " " + node->directoryName;
    bool open = ImGui::TreeNodeEx(label.c_str(), node_flags);

    if (ImGui::IsItemClicked())
    {
        selected = node;
        filter.Clear();
    }

    if (open)
    {
        // recurse for each sub directory
        for (size_t i = 0; i < node->children.size(); i++)
        {
            drawProjectNodeRecursive(node->children[i]);
        }

        ImGui::TreePop();
    }
}

InteractionType ProjectView::getInteractionTypeFromFileExtension(const std::string extension)
{
    if (extension == "mesh")
    {
        return InteractionType::Mesh;
    }
    else if (extension == "material")
    {
        return InteractionType::Material;
    }
    else if (extension == "texture")
    {
        return InteractionType::Texture2D;
    }
    else if (extension == "shader")
    {
        return InteractionType::Shader;
    }
    else
    {
        return InteractionType::Other;
    }
}