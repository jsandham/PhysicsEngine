#include "../include/ProjectTree.h"

#include "../include/FileSystemUtil.h"

#include <cassert>
#include <stack>

#include "../../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

ProjectNode::ProjectNode()
{
    mParent = nullptr;
    mDirectoryName = "";
    mDirectoryPath = "";
}

ProjectNode::ProjectNode(const std::string& path, const std::string& name)
{
    mParent = nullptr;
    mDirectoryName = name;
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

    return icon + " " + getDirectoryName();
}

std::string ProjectNode::getDirectoryName() const
{
    return mDirectoryName;
}

std::string ProjectNode::getDirectoryPath() const
{
    return mDirectoryPath;
}

std::vector<std::string> ProjectNode::getFileLabels() const
{
    return mFileLabels;
}

std::vector<std::string> ProjectNode::getFilenames() const
{
    return mFilenames;
}

std::vector<std::string> ProjectNode::getFilePaths() const
{
    return mFilePaths;
}

std::vector<std::string> ProjectNode::getFileExtensions() const
{
    return mFileExtensions;
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

std::string ProjectNode::getFilename(size_t index) const
{
    assert(index < mFilenames.size());
    return mFilenames[index];
}

std::string ProjectNode::getFilePath(size_t index) const
{
    assert(index < mFilePaths.size());
    return mFilePaths[index];
}

std::string ProjectNode::getFileExtension(size_t index) const
{
    assert(index < mFileExtensions.size());
    return mFileExtensions[index];
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
    node->mDirectoryName = name;
    node->mDirectoryPath = this->getDirectoryPath() + "\\" + name;

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
    mFilenames.push_back(name);
    mFilePaths.push_back(getDirectoryPath() + "\\" + name);
    mFileExtensions.push_back(extension);
    mFileTypes.push_back(type);
}

void ProjectNode::removeDirectory(const std::string& name)
{
    ProjectNode* nodeToDelete = nullptr;
    int index = -1;
    for (size_t i = 0; i < mChildren.size(); i++)
    {
        if (mChildren[i]->getDirectoryName() == name)
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
    std::string filepath = getDirectoryPath() + "\\" + name;
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
    mFilenames.erase(mFilenames.begin() + index);
    mFilePaths.erase(mFilePaths.begin() + index);
    mFileExtensions.erase(mFileExtensions.begin() + index);
    mFileTypes.erase(mFileTypes.begin() + index);
}


void ProjectNode::removeAllFiles()
{
    mFileLabels.clear();
    mFilenames.clear();
    mFilePaths.clear();
    mFileExtensions.clear();
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
    std::vector<std::string> filenames = getFilesInDirectory(this->getDirectoryPath(), false);
    for (size_t i = 0; i < filenames.size(); i++)
    {
        this->addFile(filenames[i]);
    }

    stack.push(this);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        // find directories that exist in the current directory
        std::vector<std::string> subDirectoryNames =
            PhysicsEditor::getDirectoriesInDirectory(current->getDirectoryPath(), false);

        // recurse for each sub directory
        for (size_t i = 0; i < subDirectoryNames.size(); i++)
        {
            ProjectNode* child = current->addDirectory(subDirectoryNames[i]);

            std::vector<std::string> filenames = PhysicsEditor::getFilesInDirectory(child->getDirectoryPath(), false);
            for (size_t j = 0; j < filenames.size(); j++)
            {
                child->addFile(filenames[j]);
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

void ProjectTree::buildProjectTree(const std::string& projectPath)
{
    deleteProjectTree();

    mRoot = new ProjectNode(projectPath + "\\data", "data");

    std::vector<std::string> filenames = getFilesInDirectory(mRoot->getDirectoryPath(), false);
    for (size_t i = 0; i < filenames.size(); i++)
    {
        mRoot->addFile(filenames[i]);
    }

    std::stack<ProjectNode*> stack;
    stack.push(mRoot);

    while (!stack.empty())
    {
        ProjectNode* current = stack.top();
        stack.pop();

        // find directories that exist in the current directory
        std::vector<std::string> subDirectoryNames =
            PhysicsEditor::getDirectoriesInDirectory(current->getDirectoryPath(), false);

        // recurse for each sub directory
        for (size_t i = 0; i < subDirectoryNames.size(); i++)
        {
            ProjectNode* child = current->addDirectory(subDirectoryNames[i]);

            std::vector<std::string> filenames = PhysicsEditor::getFilesInDirectory(child->getDirectoryPath(), false);
            for (size_t j = 0; j < filenames.size(); j++)
            {
                child->addFile(filenames[j]);
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