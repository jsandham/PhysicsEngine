#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "Guid.h"
#include "Asset.h"

namespace PhysicsEngine
{
	typedef enum TextureDimension
	{
		Tex2D = 0,
		Tex3D = 1,
		Cube = 2
	}TextureDimension;

	typedef enum TextureFormat
	{
		Depth = 0,
		RG = 1,
		RGB = 2,
		RGBA = 3
	}TextureFormat;

	class Texture : public Asset
	{
		protected:
			std::vector<unsigned char> mRawTextureData;
			int mNumChannels;
			TextureDimension mDimension;
			TextureFormat mFormat;
			GLuint mTex;
			bool mCreated;

		public:
			Texture();
			virtual ~Texture() {};

			bool isCreated() const;
			int getNumChannels() const;
			TextureDimension getDimension() const;
			TextureFormat getFormat() const;
			GLuint getNativeGraphics() const;

		protected:
			int calcNumChannels(TextureFormat format) const;
	};

	template <typename T>
	struct IsTexture { static constexpr bool value = false; };

	template <>
	struct IsTexture<Texture> { static constexpr bool value = true; };
	template <>
	struct IsAsset<Texture> { static constexpr bool value = true; };
	template <>
	struct IsAssetInternal<Texture> { static constexpr bool value = true; };
}

#endif