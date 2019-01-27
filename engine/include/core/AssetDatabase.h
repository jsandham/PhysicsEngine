#ifndef __ASSET_DATABASE_H__
#define __ASSET_DATABASE_H__

#include <map>
#include <string>

#include "Guid.h"

namespace PhysicsEngine
{
	class AssetDatabase
	{
		private:
			std::map<Guid, std::string> assetIdToFilePath;

		public:
			AssetDatabase();
			~AssetDatabase();

			void add(std::string assetPath);
			std::string get(Guid guid);
	};
}

#endif