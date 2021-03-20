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
using namespace PhysicsEngine;

ProjectView::ProjectView() : Window("Project View")
{
    root = NULL;
    selected = NULL;
}

ProjectView::~ProjectView()
{
    deleteProjectTree();
}

void ProjectView::init(EditorClipboard &clipboard)
{
}

void ProjectView::update(EditorClipboard &clipboard)
{
    if (clipboard.getProjectPath() != "")
    {
        if (root != NULL && root->directoryPath != (clipboard.getProjectPath() + "\\data") || root == NULL /*||
            editorBecameActiveThisFrame*/)
        {
            buildProjectTree(clipboard.getProjectPath());
        }
    }

    if (clipboard.getProjectPath() != "")
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

void ProjectView::drawRightPane(EditorClipboard &clipboard)
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
        if (selected != NULL)
        {
            directories = selected->children;
            files = selected->filePaths;
        }
    }

    ProjectNode *newSelection = NULL;

    // draw directories and files in right pane
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
                                     clipboard.getLibrary().getFileId(files[i]));
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(0))
        {
            clipboard.setSelectedItem(getInteractionTypeFromFileExtension(extension),
                                      clipboard.getLibrary().getFileId(files[i]));
        }

        if (!ImGui::IsMouseDown(0))
        {
            clipboard.clearDraggedItem();
        }
    }

    if (newSelection != NULL)
    {
        selected = newSelection;
    }
}

void ProjectView::deleteProjectTree()
{
    if (root == NULL)
    {
        return;
    }

    selected = NULL;

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
    root->parent = NULL;
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
    if (node == NULL)
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