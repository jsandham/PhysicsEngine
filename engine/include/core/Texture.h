#ifndef __TEXTURE_H__
#define __TEXTURE_H__

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
		Red,
		Green,
		Blue,
		Alpha,
		Depth,
		RGB,
		BGR,
		RGBA,
		BGRA
	};

	class Texture
	{
		protected:
			int width;
			int height;
			int numChannels;
			TextureDimension dimension;

		public:
			int textureId;
			// int globalIndex;

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


// #ifndef __TEXTURE_H__
// #define __TEXTURE_H__

// #include <string>
// #include <GL/glew.h>

// namespace PhysicsEngine
// {
// 	typedef enum TextureDimension
// 	{
// 		Tex2D,
// 		Tex3D,
// 		Cube
// 	};

// 	typedef enum TextureFormat
// 	{
// 		Red,
// 		Green,
// 		Blue,
// 		Alpha,
// 		Depth,
// 		RGB,
// 		BGR,
// 		RGBA,
// 		BGRA
// 	};

// 	class Texture
// 	{
// 		protected:
// 			int width;
// 			int height;
// 			int numChannels;
// 			TextureDimension dimension;

// 		public:
// 			Texture();
// 			virtual ~Texture() {};

// 			virtual void generate() = 0;
// 			virtual void destroy() = 0;
// 			virtual void bind() = 0;
// 			virtual void unbind() = 0;

// 			int getWidth() const;
// 			int getHeight() const;
// 			int getNumChannels() const;
// 			TextureDimension getDimension() const;

// 		protected:
// 			int calcNumChannels(TextureFormat format);
// 	};
// }

// #endif