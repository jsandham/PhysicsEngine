#include "../../include/views/ProjectView.h"

#include <algorithm>
#include <stack>

#include "../../include/imgui/imgui_extensions.h"

#include "core/Guid.h"
#include "core/Log.h"

using namespace PhysicsEditor;

ProjectView::ProjectView() : Window("Project View")
{
    mSelected = nullptr;
    mRightPanelSelectedPath = "";
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
        if (!mProjectTree.isEmpty() && mProjectTree.getRoot()->getDirectoryPath() != (clipboard.getProjectPath() + "\\data") || mProjectTree.isEmpty())
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
    std::vector<std::string> filenames;
    std::vector<std::string> filePaths;
    std::vector<std::string> fileExtensions;
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

            if (mFilter.PassFilter(current->getDirectoryName().c_str()))
            {
                directories.push_back(current);
            }

            for (size_t j = 0; j < current->getFileCount(); j++)
            {
                if (mFilter.PassFilter(current->getFilename(j).c_str()))
                {
                    fileLabels.push_back(current->getFileLabel(j));
                    filenames.push_back(current->getFilename(j));
                    filePaths.push_back(current->getFilePath(j));
                    fileExtensions.push_back(current->getFileExtension(j));
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
            filenames = mSelected->getFilenames();
            filePaths = mSelected->getFilePaths();
            fileExtensions = mSelected->getFileExtensions();
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
            clipboard.mSelectedPath = directories[i]->getDirectoryPath();
        }
    }

    // draw files in right pane
    for (size_t i = 0; i < filePaths.size(); i++)
    {
        if (ImGui::Selectable(fileLabels[i].c_str(), filePaths[i] == mRightPanelSelectedPath))
        {
            mRightPanelSelectedPath = filePaths[i];
        }

        if (ImGui::IsItemHovered())
        {
            if (ImGui::IsMouseClicked(0))
            {
                clipboard.mDraggedType = fileTypes[i];
                clipboard.mDraggedPath = filePaths[i];
                clipboard.mDraggedId = clipboard.getLibrary().getId(clipboard.mDraggedPath);
            }

            if (ImGui::IsMouseReleased(0))
            {
                clipboard.mSelectedType = fileTypes[i];
                clipboard.mSelectedPath = filePaths[i];
                clipboard.mSelectedId = clipboard.getLibrary().getId(clipboard.mSelectedPath);
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
                std::string folderPath = mSelected->getDirectoryPath() + "\\" + foldername;
                if (PhysicsEditor::createDirectory(folderPath))
                {
                    mSelected->addDirectory(foldername);
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Material"))
            {
                size_t count = clipboard.getWorld()->getNumberOfAssets<PhysicsEngine::Material>();
                std::string filename = ("NewMaterial(" + std::to_string(count) + ").material");
                std::string filepath = mSelected->getDirectoryPath() + "\\" + filename;

                PhysicsEngine::Material* material = clipboard.getWorld()->createAsset<PhysicsEngine::Material>();
                material->writeToYAML(filepath);

                mSelected->addFile(filename);
            }

            ImGui::EndMenu();
        }
        
        if (ImGui::MenuItem("Delete", nullptr, false, !clipboard.getSelectedPath().empty()))
        {
            if (clipboard.mSelectedType == InteractionType::Folder)
            {
                std::string folderpath = clipboard.getSelectedPath();
                if (PhysicsEditor::deleteDirectory(folderpath))
                {
                    clipboard.clearSelectedItem();

                    mSelected->removeDirectory(folderpath.substr(folderpath.find_last_of("\\") + 1));

                    if (folderpath == mRightPanelSelectedPath)
                    {
                        mRightPanelSelectedPath = "";
                    }
                }
            }
            else
            {
                std::string filepath = clipboard.getSelectedPath();
                if (PhysicsEditor::deleteFile(filepath))
                {
                    clipboard.clearSelectedItem();

                    mSelected->removeFile(filepath.substr(filepath.find_last_of("\\") + 1));
                    
                    if (filepath == mRightPanelSelectedPath)
                    {
                        mRightPanelSelectedPath = "";
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