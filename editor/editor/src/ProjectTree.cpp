#include "../include/ProjectTree.h"

#include <cassert>
#include <stack>

#include "../../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

ProjectNode::ProjectNode()
{
    mParent = nullptr;
    mDirectoryPath = std::filesystem::path();
}

ProjectNode::ProjectNode(const std::filesystem::path& path)
{
    mParent = nullptr;
    mDirectoryPath = path;
}

ProjectNode::~ProjectNode()
{

}

size_t ProjectNode::getFileCount() const
{
    return mFilePaths.size();
}

std::string ProjectNode::getDirectoryLabel() const
{
    std::string icon = std::string(ICON_FA_FOLDER);
    if (getChildCount() == 0)
    {
        if (getFileCount() == 0)
        {
            icon = std::string(ICON_FA_FOLDER_O);
        }
    }

    return icon + " " + getDirectoryPath().filename().string();
}

std::filesystem::path ProjectNode::getDirectoryPath() const
{
    return mDirectoryPath;
}

std::vector<std::string> ProjectNode::getFileLabels() const
{
    return mFileLabels;
}

std::vector<std::filesystem::path> ProjectNode::getFilePaths() const
{
    return mFilePaths;
}

std::vector<InteractionType> ProjectNode::getFileTypes() const
{
    return mFileTypes;
}

std::string ProjectNode::getFileLabel(size_t index) const
{
    assert(index < mFileLabels.size());
    return mFileLabels[index];
}

std::filesystem::path ProjectNode::getFilePath(size_t index) const
{
    assert(index < mFilePaths.size());
    return mFilePaths[index];
}

InteractionType ProjectNode::getFileType(size_t index) const
{
    assert(index < mFileTypes.size());
    return mFileTypes[index];
}

ProjectNode* ProjectNode::addDirectory(const std::string& name)
{
    ProjectNode* node = new ProjectNode();
    node->mParent = this;
    node->mDirectoryPath = getDirectoryPath() / name;

    mChildren.push_back(node);

    return node;
}

void ProjectNode::addFile(const std::string& name)
{
    std::string extension = name.substr(name.find_last_of(".") + 1);
    std::string label = std::string(ICON_FA_FILE);
    InteractionType type = InteractionType::None;
    if (extension == "material") 
    {
        label = std::string(ICON_FA_MAXCDN);
        type = InteractionType::Material;
    }
    else if (extension == "mesh")
    {
        label = std::string(ICON_FA_CODEPEN);
        type = InteractionType::Mesh;
    }
    else if (extension == "texture")
    {
        label = std::string(ICON_FA_FILE_IMAGE_O);
        type = InteractionType::Texture2D;
    }
    else if (extension == "shader")
    {
        label = std::string(ICON_FA_AREA_CHART);
        type = InteractionType::Shader;
    }

    mFileLabels.push_back(label + " " + name);
    mFilePaths.push_back(getDirectoryPath() / name);
    mFileTypes.push_back(type);
}

void ProjectNode::removeDirectory(const std::string& name)
{
    ProjectNode* nodeToDelete = nullptr;
    int index = -1;
    for (size_t i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getDirectoryPath().filename().string() == name)
        {
            nodeToDelete = mChildren[i];
            index = i;
        }
    }

    if (nodeToDelete == nullptr)
    {
        return;
    }

    mChildren.erase(mChildren.begin() + index);

    std::stack<ProjectNode*> stack;
    stack.push(nodeToDelete);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        for (size_t i = 0; i < current->mChildren.size(); i++)
        {
            stack.push(current->mChildren[i]);
        }

        delete current;
    }
}

void ProjectNode::removeFile(const std::string& name)
{
    std::filesystem::path filepath = getDirectoryPath() / name;
    int index = -1;
    for (size_t i = 0; i < mFilePaths.size(); i++)
    {
        if (mFilePaths[i] == filepath)
        {
            index = (int)i;
            break;
        }
    }

    if (index == -1)
    {
        return;
    }

    mFileLabels.erase(mFileLabels.begin() + index);
    mFilePaths.erase(mFilePaths.begin() + index);
    mFileTypes.erase(mFileTypes.begin() + index);
}


void ProjectNode::removeAllFiles()
{
    mFileLabels.clear();
    mFilePaths.clear();
    mFileTypes.clear();
}

void ProjectNode::rebuild()
{
    std::stack<ProjectNode*> stack;

    // delete all node children
    for (size_t i = 0; i < this->getChildCount(); i++)
    {
        stack.push(this->getChild(i));
    }

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        for (size_t i = 0; i < current->getChildCount(); i++)
        {
            stack.push(current->getChild(i));
        }

        delete current;
    }

    // delete all node files
    this->removeAllFiles();

    // rebuild node
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(getDirectoryPath()))
    {
        if (std::filesystem::is_regular_file(entry))
        {
            this->addFile(entry.path().filename().string());
        }
    }

    stack.push(this);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        // find directories that exist in the current directory
        std::vector<std::filesystem::path> subDirectoryNames;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(current->getDirectoryPath()))
        {
            if (std::filesystem::is_directory(entry))
            {
                subDirectoryNames.push_back(entry.path().filename());
            }
        }

        // recurse for each sub directory
        for (size_t i = 0; i < subDirectoryNames.size(); i++)
        {
            ProjectNode* child = current->addDirectory(subDirectoryNames[i].string());

            for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(child->getDirectoryPath()))
            {
                if (std::filesystem::is_regular_file(entry))
                {
                    child->addFile(entry.path().filename().string());
                }
            }

            stack.push(child);
        }
    }
}

ProjectNode* ProjectNode::getChild(size_t index)
{
    assert(index < mChildren.size());
    return mChildren[index];
}

std::vector<ProjectNode*> ProjectNode::getChildren()
{
    return mChildren;
}

size_t ProjectNode::getChildCount() const
{
    return mChildren.size();
}

ProjectTree::ProjectTree()
{
    mRoot = nullptr;
}

ProjectTree::~ProjectTree()
{
    deleteProjectTree();
}

bool ProjectTree::isEmpty() const
{
    return mRoot == nullptr;
}

ProjectNode* ProjectTree::getRoot()
{
    return mRoot;
}

void ProjectTree::buildProjectTree(const std::filesystem::path& projectPath)
{
    deleteProjectTree();

    mRoot = new ProjectNode(projectPath / "data");

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(mRoot->getDirectoryPath()))
    {
        if (std::filesystem::is_regular_file(entry))
        {
            mRoot->addFile(entry.path().filename().string());
        }
    }

    std::stack<ProjectNode*> stack;
    stack.push(mRoot);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        // find directories that exist in the current directory
        std::vector<std::filesystem::path> subDirectoryNames;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(current->getDirectoryPath()))
        {
            if (std::filesystem::is_directory(entry))
            {
                subDirectoryNames.push_back(entry.path().filename());
            }
        }

        // recurse for each sub directory
        for (size_t i = 0; i < subDirectoryNames.size(); i++)
        {
            ProjectNode* child = current->addDirectory(subDirectoryNames[i].string());

            for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(child->getDirectoryPath()))
            {
                if (std::filesystem::is_regular_file(entry))
                {
                    child->addFile(entry.path().filename().string());
                }
            }

            stack.push(child);
        }
    }
}

void ProjectTree::deleteProjectTree()
{
    if (mRoot == nullptr)
    {
        return;
    }

    std::stack<ProjectNode*> stack;
    stack.push(mRoot);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        for (size_t i = 0; i < current->getChildCount(); i++)
        {
            stack.push(current->getChild(i));
        }

        delete current;
    }
}