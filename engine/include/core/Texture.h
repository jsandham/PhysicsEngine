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
			int numChannels;
			TextureDimension dimension;
			TextureFormat format;

			std::vector<unsigned char> rawTextureData;

		public:
			GLuint tex;

		public:
			Texture();
			virtual ~Texture() {};

			int getNumChannels() const;
			TextureDimension getDimension() const;
			TextureFormat getFormat() const;

		protected:
			int calcNumChannels(TextureFormat format);
	};

	template <typename T>
	struct IsTexture { static bool value; };

	template <typename T>
	bool IsTexture<T>::value = false;

	template<>
	bool IsTexture<Texture>::value = true;
	template<>
	bool IsAsset<Texture>::value = true;
}

#endif