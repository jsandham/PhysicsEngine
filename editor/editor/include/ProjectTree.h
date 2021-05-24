#ifndef PROJECT_TREE_H__
#define PROJECT_TREE_H__

#include <string>
#include <vector>
#include <filesystem>

#include "InteractionType.h"

namespace PhysicsEditor
{
    class ProjectNode
    {
    private:
        ProjectNode* mParent;
        std::vector<ProjectNode*> mChildren;
        std::string mDirectoryLabel;
        std::filesystem::path mDirectoryPath;
        std::vector<std::string> mFileLabels;
        std::vector<std::filesystem::path> mFilePaths;
        std::vector<InteractionType> mFileTypes;
        
    public:
        ProjectNode();
        ProjectNode(const std::filesystem::path& path);
        ~ProjectNode();

        size_t getFileCount() const;

        std::string getDirectoryLabel() const;
        std::filesystem::path getDirectoryPath() const;

        std::vector<std::string> getFileLabels() const;
        std::vector<std::filesystem::path> getFilePaths() const;
        std::vector<InteractionType> getFileTypes() const;

        std::string getFileLabel(size_t index) const;
        std::filesystem::path getFilePath(size_t index) const;
        InteractionType getFileType(size_t index) const;

        ProjectNode* addDirectory(const std::string& name);
        void addFile(const std::string& name);
        void removeDirectory(const std::string& name);
        void removeFile(const std::string& name);
        void removeAllFiles();
        void rebuild();

        size_t getChildCount() const;
        ProjectNode* getChild(size_t index);
        std::vector<ProjectNode*> getChildren();
    };

    class ProjectTree
    {
    private:
        ProjectNode* mRoot;

    public:
        ProjectTree();
        ~ProjectTree();

        bool isEmpty() const;
        ProjectNode* getRoot();

        void buildProjectTree(const std::filesystem::path& projectPath);
        void deleteProjectTree();
    };
}

#endif