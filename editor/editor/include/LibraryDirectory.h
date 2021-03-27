#ifndef __LIBRARY_DIRECTORY_H__
#define __LIBRARY_DIRECTORY_H__

#include <map>
#include <string>
#include <vector>

#include "FileSystemUtil.h"

#include "FileWatcher.h"

#include "core/World.h"
#include "core/Guid.h"
#include "core/Log.h"

namespace PhysicsEditor
{
class LibraryDirectory;

class LibraryDirectoryListener : public FW::FileWatchListener
{
  private:
    LibraryDirectory *directory;

  public:
    LibraryDirectoryListener();
    void registerDirectory(LibraryDirectory *directory);
    void handleFileAction(FW::WatchID watchid, const FW::String &dir, const FW::String &filename, FW::Action action);
};

class LibraryDirectory
{
  private:
    // data directory path
    std::string mDataPath;

    // filepath to id map
    std::map<const std::string, PhysicsEngine::Guid> filePathToId;

    // buffer of added/modified project file paths
    std::vector<std::string> mBuffer;

    // file watcher listener object
    LibraryDirectoryListener mListener;

    // create the file watcher object
    FW::FileWatcher mFileWatcher;

    // current watch id
    FW::WatchID mWatchID;

  public:
    LibraryDirectory();
    ~LibraryDirectory();
    LibraryDirectory(const LibraryDirectory& other) = delete;
    LibraryDirectory& operator=(const LibraryDirectory& other) = delete;

    void watch(const std::string& projectPath);
    void update();
    void addToBuffer(const std::string& filePath);
    void loadQueuedAssetsIntoWorld(PhysicsEngine::World *world);

    PhysicsEngine::Guid getFileId(const std::string &filePath) const;
};
} // namespace PhysicsEditor

#endif
