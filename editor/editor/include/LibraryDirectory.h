#ifndef __LIBRARY_DIRECTORY_H__
#define __LIBRARY_DIRECTORY_H__

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "FileSystemUtil.h"
#include "EditorFileIO.h"

#include "FileWatcher.h"

#include "core/Log.h"
#include "core/Guid.h"

namespace PhysicsEditor
{
	class LibraryDirectory;

	class LibraryDirectoryListener : public FW::FileWatchListener
	{
		private:
			LibraryDirectory* directory;

		public:
			LibraryDirectoryListener();
			void registerDirectory(LibraryDirectory* directory);
			void handleFileAction(FW::WatchID watchid, const FW::String& dir, const FW::String& filename, FW::Action action);
	};

	class LibraryDirectory 
	{
		private:
			// data directory path
			std::string mDataPath;

			// library directory path
			std::string mLibraryPath;

			// buffer of added/modified library file paths
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

			void watch(std::string projectPath);
			void update();
			void loadQueuedAssetsIntoWorld(PhysicsEngine::World* world);
			void generateBinaryLibraryFile(std::string filePath);
	};
}

#endif
