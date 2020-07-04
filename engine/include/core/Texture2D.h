#ifndef __TEXTURE2D_H__
#define __TEXTURE2D_H__

#include <vector>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{	
#pragma pack(push, 1)
	struct Texture2DHeader
	{
		Guid mTextureId;
		int mWidth;
		int mHeight;
		int mNumChannels;
		int mDimension;
		int mFormat;
		size_t mTextureSize;
	};
#pragma pack(pop)

	class Texture2D : public Texture
	{
		private:
			int mWidth;
			int mHeight;

		public:
			Texture2D();
			Texture2D(std::vector<char> data);
			Texture2D(int width, int height);
			Texture2D(int width, int height, TextureFormat format);
			~Texture2D();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid assetId) const;
			void deserialize(std::vector<char> data);

			void load(const std::string& filepath);

			int getWidth() const;
			int getHeight() const;

			void redefine(int width, int height, TextureFormat format);

			std::vector<unsigned char> getRawTextureData() const;
			std::vector<Color32> getPixels() const;
			Color32 getPixel(int x, int y) const;

			void setRawTextureData(std::vector<unsigned char> data, int width, int height, TextureFormat format);
			void setPixels(std::vector<Color32> colors);
			void setPixel(int x, int y, Color32 color);

			void create();
			void destroy();
			void readPixels();
			void apply();
	};

	template <>
	const int AssetType<Texture2D>::type = 1;

	template <typename T>
	struct IsTexture2D { static const bool value; };

	template <typename T>
	const bool IsTexture2D<T>::value = false;

	template<>
	const bool IsTexture2D<Texture2D>::value = true;
	template<>
	const bool IsTexture<Texture2D>::value = true;
	template<>
	const bool IsAsset<Texture2D>::value = true;
	template<>
	const bool IsAssetInternal<Texture2D>::value = true;
}

#endif