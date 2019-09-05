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
			std::unordered_set<std::string> trackedFilesInProject;
			std::map<PhysicsEngine::Guid, std::string> idToTrackedFilePath;

		public:
			LibraryDirectory();
			~LibraryDirectory();

			void update(std::string projectPath);
			void createBinaryAssetInLibrary(std::string filePath, std::string extension);
			void createBinarySceneInLibrary(std::string filePath);

			std::string getPathToBinarySceneOrAsset(PhysicsEngine::Guid id);
	};
}

#endif
