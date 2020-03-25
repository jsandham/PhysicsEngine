#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Guid.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct AssetHeader
	{
		unsigned short mFileType;
		unsigned int mFileSize;
	};
#pragma pack(pop)
	
	class World;

	template <typename T>
	struct AssetType { static const int type; };

	template <typename T>
	const int AssetType<T>::type = -1;

	class Asset
	{
		protected:
			Guid mAssetId;

		public:
			Asset();
			virtual ~Asset() = 0;

			virtual std::vector<char> serialize() const = 0;
			virtual std::vector<char> serialize(Guid assetId) const = 0;
			virtual void deserialize(std::vector<char> data) = 0;

			Guid getId() const;

		private:
			friend class World;
	};

	template <typename T>
	struct IsAsset { static const bool value; };

	template <typename T>
	const bool IsAsset<T>::value = false;

	template<>
	const bool IsAsset<Asset>::value = true;
}

#endif