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
		Tex2D,
		Tex3D,
		Cube
	}TextureDimension;

	typedef enum TextureFormat
	{
		Depth,
		RG,
		RGB,
		RGBA
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
	struct IsTexture { static const bool value; };

	template <typename T>
	const bool IsTexture<T>::value = false;

	template<>
	const bool IsTexture<Texture>::value = true;
	template<>
	const bool IsAsset<Texture>::value = true;
}

#endif