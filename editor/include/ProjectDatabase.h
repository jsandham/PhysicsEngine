#ifndef PROJECT_DATABASE_H__
#define PROJECT_DATABASE_H__

#include <map>
#include <string>
#include <vector>
#include <filesystem>

#include <efsw/efsw.hpp>

#include "core/World.h"
#include "core/Guid.h"
#include "core/Log.h"

#include "EditorClipboard.h"

namespace PhysicsEditor
{
    class DirectoryListener : public efsw::FileWatchListener 
    {
    public:
        void handleFileAction(efsw::WatchID watchid, const std::string& dir, const std::string& filename, efsw::Action action, std::string oldFilename) override;
    };

    struct Action
    {
        std::filesystem::path mPath;
        efsw::Action mAction;
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

        static std::queue<Action> mActionQueue;

        // file watcher listener object
        static DirectoryListener mListener;

        // create the file watcher object
        static efsw::FileWatcher mFileWatcher;

        // current watch id
        static efsw::WatchID mWatchID;

    public:
        static void watch(const std::filesystem::path& projectPath);
        static void update(PhysicsEngine::World* world);
        
        static void queueFileAction(Action action);
        static void addFile(const std::filesystem::path& path, PhysicsEngine::World* world);
        static void deleteFile(const std::filesystem::path& path, PhysicsEngine::World* world);

        static void createDirectory(const std::filesystem::path& parentPath);
        static void createShaderFile(const std::filesystem::path& parentPath);
        static void createCubemapFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void createMaterialFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void createRenderTextureFile(PhysicsEngine::World* world, const std::filesystem::path& parentPath);
        static void rename(const std::filesystem::path& oldPath, const std::filesystem::path& newPath);
        static void remove_all(const std::filesystem::path& path);
        static void remove(const std::filesystem::path& path);
        static PhysicsEngine::Guid getGuid(const std::filesystem::path& filePath);
        static std::filesystem::path getFilePath(const PhysicsEngine::Guid& guid);

        static void newProject(Clipboard& clipboard, const std::filesystem::path& projectPath);
        static void openProject(Clipboard& clipboard, const std::filesystem::path& projectPath);
        static void saveProject(Clipboard& clipboard);

        static void newScene(Clipboard& clipboard, const std::string& sceneName);
        static void openScene(Clipboard& clipboard, const std::filesystem::path& scenePath);
        static void saveScene(Clipboard& clipboard, const std::filesystem::path& scenePath);

        static void populateScene(Clipboard& clipboard);
    };
} // namespace PhysicsEditor

#endif