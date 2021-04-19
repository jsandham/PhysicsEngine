#ifndef PROJECT_TREE_H__
#define PROJECT_TREE_H__

#include <string>
#include <vector>

#include "InteractionType.h"

namespace PhysicsEditor
{
    class ProjectNode
    {
    private:
        ProjectNode* mParent;
        std::vector<ProjectNode*> mChildren;
        std::string mDirectoryLabel;
        std::string mDirectoryName;
        std::string mDirectoryPath;
        std::vector<std::string> mFileLabels;
        std::vector<std::string> mFilenames;
        std::vector<std::string> mFilePaths;
        std::vector<std::string> mFileExtensions;
        std::vector<InteractionType> mFileTypes;
        
    public:
        ProjectNode();
        ProjectNode(const std::string& path, const std::string& name);
        ~ProjectNode();

        size_t getFileCount() const;

        std::string getDirectoryLabel() const;
        std::string getDirectoryName() const;
        std::string getDirectoryPath() const;

        std::vector<std::string> getFileLabels() const;
        std::vector<std::string> getFilenames() const;
        std::vector<std::string> getFilePaths() const;
        std::vector<std::string> getFileExtensions() const;
        std::vector<InteractionType> getFileTypes() const;

        std::string getFileLabel(size_t index) const;
        std::string getFilename(size_t index) const;
        std::string getFilePath(size_t index) const;
        std::string getFileExtension(size_t index) const;
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

        void buildProjectTree(const std::string& projectPath);
        void deleteProjectTree();
    };
}

#endif