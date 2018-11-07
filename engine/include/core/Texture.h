#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include "Guid.h"
#include "Asset.h"

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	typedef enum TextureDimension
	{
		Tex2D,
		Tex3D,
		Cube
	};

	typedef enum TextureFormat
	{
		Depth,
		RG,
		RGB,
		RGBA
	};

	class Texture : public Asset
	{
		protected:
			int width;
			int height;
			int numChannels;
			TextureDimension dimension;

		public:
			GLHandle handle;

		public:
			Texture();
			virtual ~Texture() {};

			int getWidth() const;
			int getHeight() const;
			int getNumChannels() const;
			TextureDimension getDimension() const;

		protected:
			int calcNumChannels(TextureFormat format);
	};
}

#endif