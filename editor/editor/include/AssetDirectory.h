#ifndef __ASSET_DIRECTORY_H__
#define __ASSET_DIRECTORY_H__

#include <string>
#include <unordered_set>

namespace PhysicsEditor
{
	class AssetDirectory
	{
		private:
			std::string currentProjectPath;
			std::unordered_set<std::string> directory;

		public:
			AssetDirectory();
			~AssetDirectory();

			void update(std::string projectPath);
			void createBinaryAssetInLibrary(std::string filePath, std::string extension);
	};
}

#endif
