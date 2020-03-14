#ifndef __CUBEMAP_H__
#define __CUBEMAP_H__

#include <vector>
#include <string>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct CubemapHeader
	{
		Guid textureId;
		int width;
		int numChannels;
		int dimension;
		int format;
		size_t textureSize;
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
			int width;

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
			std::vector<Color> getPixels(CubemapFace face) const;
			Color getPixel(CubemapFace face, int x, int y) const;
			
			void setRawCubemapData(std::vector<unsigned char> data);
			void setPixels(CubemapFace face, int x, int y, Color color);
			void setPixel(CubemapFace face, int x, int y, Color color);

			void create();
			void destroy();
			void readPixels();
			void apply();
	};

	template <>
	const int AssetType<Cubemap>::type = 3;

	template <typename T>
	struct IsCubemap { static bool value; };

	template <typename T>
	bool IsCubemap<T>::value = false;

	template<>
	bool IsCubemap<Cubemap>::value = true;
	template<>
	bool IsTexture<Cubemap>::value = true;
	template<>
	bool IsAsset<Cubemap>::value = true;
}

#endif