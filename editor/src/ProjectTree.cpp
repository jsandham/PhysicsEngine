#include "../include/ProjectTree.h"

#include <cassert>
#include <queue>

#include "../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

ProjectNode::ProjectNode()
{
    mParent = nullptr;
    mDirectoryPath = std::filesystem::path();

    mFileLabels.reserve(200);
    mFilePaths.reserve(200);
    mFileTypes.reserve(200);
}

ProjectNode::ProjectNode(const std::filesystem::path& path)
{
    mParent = nullptr;
    mDirectoryPath = path;

    mFileLabels.reserve(200);
    mFilePaths.reserve(200);
    mFileTypes.reserve(200);
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

ProjectNode* ProjectNode::addDirectory(const std::filesystem::path& path)
{
    ProjectNode* node = new ProjectNode();
    node->mParent = this;
    node->mDirectoryPath = path;

    mChildren.push_back(node);

    return node;
}

void ProjectNode::addFile(const std::filesystem::path& path)
{
    std::string filename = path.filename().string();
    std::string extension = filename.substr(filename.find_last_of(".") + 1);

    std::string label;
    InteractionType type;
    if(extension.length() >= 1)
    {
        if (extension[0] == 's')
        {
            if (extension == "scene")
            {
                label = std::string(ICON_FA_MAXCDN);
                type = InteractionType::Scene;
            }
            else if (extension == "shader")
            {
                label = std::string(ICON_FA_AREA_CHART);
                type = InteractionType::Shader;
            }
        }
        else if(extension[0] == 'm')
        {
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
        }
        else if (extension == "texture")
        {
            label = std::string(ICON_FA_FILE_IMAGE_O);
            type = InteractionType::Texture2D;
        }
        else if (extension == "rendertexture")
        {
            label = std::string(ICON_FA_AREA_CHART);
            type = InteractionType::RenderTexture;
        }
        else if (extension == "cubemap")
        {
            label = std::string(ICON_FA_AREA_CHART);
            type = InteractionType::Cubemap;
        }
        else
        {
            label = std::string(ICON_FA_FILE);
            type = InteractionType::File;
        }
    }
    else
    {
        label = std::string(ICON_FA_FILE);
        type = InteractionType::File;
    }

    mFileLabels.push_back(label + " " + filename);
    mFilePaths.push_back(path);
    mFileTypes.push_back(type);    
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

    std::queue<ProjectNode*> queue;
    queue.push(mRoot);

    while (!queue.empty())
    {
        ProjectNode* current = queue.front();
        queue.pop();

        for (const auto& entry : std::filesystem::directory_iterator(current->getDirectoryPath()))
        {
            if (entry.is_regular_file())
            {
                current->addFile(entry.path());
            }
            else if (entry.is_directory())
            {
                ProjectNode* child = current->addDirectory(entry.path());
                queue.push(child);
            }
        }
    }
}

void ProjectTree::deleteProjectTree()
{
    if (mRoot != nullptr)
    {
        std::queue<ProjectNode*> queue;
        queue.push(mRoot);

        while (!queue.empty())
        {
            ProjectNode* current = queue.front();
            queue.pop();

            for (size_t i = 0; i < current->getChildCount(); i++)
            {
                queue.push(current->getChild(i));
            }

            delete current;
        }
    }
}