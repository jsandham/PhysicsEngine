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
			std::string currentProjectPath;

		public:
			LibraryDirectory();
			~LibraryDirectory();

			void load(std::string projectPath);
			void update(std::string projectPath);

			LibraryCache getLibraryCache() const;

			static bool isFileExtensionTracked(std::string extension);
			static bool createMetaFile(std::string metaFilePath);
			static PhysicsEngine::Guid findGuidFromMetaFilePath(std::string metaFilePath);
	};
}

#endif
