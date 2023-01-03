#ifndef PROJECT_DATABASE_H__
#define PROJECT_DATABASE_H__

#include <map>
#include <string>
#include <vector>
#include <filesystem>

#include "FileWatcher.h"

#include "core/World.h"
#include "core/Guid.h"
#include "core/Log.h"

namespace PhysicsEditor
{
    class DirectoryListener : public FW::FileWatchListener
    {
    public:
        void handleFileAction(FW::WatchID watchid, const FW::String& dir, const FW::String& filename, FW::Action action);
    };

    class ProjectDatabase
    {
    private:
        // data directory path
        static std::filesystem::path mDataPath;

        // filepath to id map
        static std::map<const std::filesystem::path, PhysicsEngine::Guid> mFilePathToId;

        // filepath to id map
        static std::map<const PhysicsEngine::Guid, std::filesystem::path> mIdToFilePath;

        // buffer of added/modified project file paths
        static std::vector<std::filesystem::path> mAddBuffer;

        // buffer of deleted project file paths
        static std::vector<std::filesystem::path> mDeleteBuffer;

        // file watcher listener object
        static DirectoryListener mListener;

        // create the file watcher object
        static FW::FileWatcher mFileWatcher;

        // current watch id
        static FW::WatchID mWatchID;

    public:
        static void watch(const std::filesystem::path& projectPath);
        static void update(PhysicsEngine::World* world); //refresh??
        static void fileAddedToProject(const std::filesystem::path& filePath);
        static void fileDeletedFromProject(const std::filesystem::path& filePath);

        static bool isAssetYamlExtension(const std::string& extension);
        static bool isTextureExtension(const std::string& extension);
        static bool isMeshExtension(const std::string& extension);
        static bool isShaderExtension(const std::string& extension);


        static void createDirectory(const std::filesystem::path& parentPath);
        static void createShaderFile(const std::filesystem::path& parentPath);
        static void createCubemapFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void createMaterialFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void createSpriteFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void createRenderTextureFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void move(std::filesystem::path& oldPath, std::filesystem::path& newPath);
        static void rename(std::filesystem::path& oldPath, std::string& newFilename);
        static PhysicsEngine::Guid getGuid(const std::filesystem::path& filePath);
        static std::filesystem::path getFilePath(const PhysicsEngine::Guid& guid);
    };
} // namespace PhysicsEditor

#endif