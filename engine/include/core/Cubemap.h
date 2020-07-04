#ifndef __CUBEMAP_H__
#define __CUBEMAP_H__

#include <vector>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct CubemapHeader
	{
		Guid mTextureId;
		int mWidth;
		int mNumChannels;
		int mDimension;
		int mFormat;
		size_t mTextureSize;
	};
#pragma pack(pop)

	typedef enum CubemapFace
	{
		PositiveX,
		NegativeX,
		PositiveY,
		NegativeY,
		PositiveZ,
		NegativeZ
	}CubemapFace;

	class Cubemap : public Texture
	{
		private:
			int mWidth;

		public:
			Cubemap();
			Cubemap(std::vector<char> data);
			Cubemap(int width);
			Cubemap(int width, TextureFormat format);
			Cubemap(int width, int height, TextureFormat format);
			~Cubemap();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			int getWidth() const;

			std::vector<unsigned char> getRawCubemapData() const;
			std::vector<Color32> getPixels(CubemapFace face) const;
			Color32 getPixel(CubemapFace face, int x, int y) const;
			
			void setRawCubemapData(std::vector<unsigned char> data);
			void setPixels(CubemapFace face, int x, int y, Color32 color);
			void setPixel(CubemapFace face, int x, int y, Color32 color);

			void create();
			void destroy();
			void readPixels();
			void apply();
	};

	template <>
	const int AssetType<Cubemap>::type = 3;

	template <typename T>
	struct IsCubemap { static const bool value; };

	template <typename T>
	const bool IsCubemap<T>::value = false;

	template<>
	const bool IsCubemap<Cubemap>::value = true;
	template<>
	const bool IsTexture<Cubemap>::value = true;
	template<>
	const bool IsAsset<Cubemap>::value = true;
	template<>
	const bool IsAssetInternal<Cubemap>::value = true;
}

#endif