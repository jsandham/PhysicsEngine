#ifndef __LIBRARY_DIRECTORY_H__
#define __LIBRARY_DIRECTORY_H__

#include <string>
#include <unordered_set>
#include <map>

#include "core/Guid.h"

namespace PhysicsEditor
{
	typedef struct FileInfo
	{
		std::string filePath;
		std::string fileExtension;
		PhysicsEngine::Guid id;
		std::string createTime;
		std::string accessTime;
		std::string writeTime;
	}FileInfo;

	class LibraryDirectory 
	{
		private:
			std::string currentProjectPath;
			std::map<std::string, FileInfo> filePathToFileInfo;

		public:
			LibraryDirectory();
			~LibraryDirectory();

			void update(std::string projectPath);

			std::map<std::string, FileInfo> getTrackedFilesInProject() const;
		
		private:
			bool load();
			bool save();
			bool writeAssetToLibrary(FileInfo fileInfo);
			bool writeSceneToLibrary(FileInfo fileInfo);
	};
}

#endif
