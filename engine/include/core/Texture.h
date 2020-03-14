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
			std::vector<unsigned char> rawTextureData;
			int numChannels;
			TextureDimension dimension;
			TextureFormat format;
			GLuint tex;
			bool created;

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
	struct IsTexture { static bool value; };

	template <typename T>
	bool IsTexture<T>::value = false;

	template<>
	bool IsTexture<Texture>::value = true;
	template<>
	bool IsAsset<Texture>::value = true;
}

#endif