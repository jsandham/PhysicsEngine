#ifndef __LIBRARY_DIRECTORY_H__
#define __LIBRARY_DIRECTORY_H__

#include <string>
#include <unordered_set>
#include <map>

#include "LibraryCache.h"

#include "core/Guid.h"

namespace PhysicsEditor
{
	class LibraryDirectory 
	{
		private:
			LibraryCache libraryCache; // rename to trackedFileCache or just fileCache?
			std::map<std::string, PhysicsEngine::Guid> filePathToId;
			std::map<PhysicsEngine::Guid, std::string> idToFilePath;
		
			std::string currentProjectPath;

		public:
			LibraryDirectory();
			~LibraryDirectory();

			void load(std::string projectPath);
			void update(std::string projectPath);

			const LibraryCache& getLibraryCache() const;
			PhysicsEngine::Guid getFileId(std::string filePath) const;
			std::string getFilePath(PhysicsEngine::Guid id) const;

			bool isFileExtensionTracked(std::string extension) const;
			bool createMetaFile(std::string metaFilePath) const;
			PhysicsEngine::Guid findGuidFromMetaFilePath(std::string metaFilePath) const;
	};
}

#endif
