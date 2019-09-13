#ifndef __LIBRARY_DIRECTORY_H__
#define __LIBRARY_DIRECTORY_H__

#include <string>
#include <unordered_set>
#include <map>

#include "core/Guid.h"

namespace PhysicsEditor
{
	class LibraryDirectory 
	{
		private:
			std::string currentProjectPath;
			std::map<std::string, PhysicsEngine::Guid> filePathToId;

		public:
			LibraryDirectory();
			~LibraryDirectory();

			void update(std::string projectPath);

			std::map<std::string, PhysicsEngine::Guid> getTrackedFilesInProject() const;
		
		private:
			bool load();
			bool save();
			bool createBinaryAssetInLibrary(std::string filePath, PhysicsEngine::Guid id, std::string extension);
			bool createBinarySceneInLibrary(std::string filePath, PhysicsEngine::Guid id);
	};
}

#endif
