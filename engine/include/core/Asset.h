#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Guid.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct AssetBundleHeader
	{
		unsigned int numberOfShaders;
		unsigned int numberOfTextures;
		unsigned int numberOfMaterials;
		unsigned int numberOfMeshes;
		unsigned int numberOfGMeshes;
	};
#pragma pack(pop)

	struct AssetBundle
	{
		std::string filepath;
	};
	
	class World;

	class Asset
	{
		protected:
			World* world;

		public:
			Guid assetId;

		public:
			Asset();
			virtual ~Asset() = 0;

			void setManager(World* world);

			template <typename T>
			static int getInstanceType()
			{
				// static variables only run the first time the function is called
			    static int id = nextValue();
			    return id;
			}

		private:
			static int nextValue()
			{
				// static variables only run the first time the function is called
			    static int id = 0;
			    int result = id;
			    ++id;
			    return result;
			}

	};
}

#endif