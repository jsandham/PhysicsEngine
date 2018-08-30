#ifndef __CUBEMAP_H__
#define __CUBEMAP_H__

#include <vector>
#include <string>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
	typedef enum CubemapFace
	{
		PositiveX,
		NegativeX,
		PositiveY,
		NegativeY,
		PositiveZ,
		NegativeZ
	};

	class Cubemap : public Texture
	{
		private:
			std::vector<unsigned char> rawCubemapData;

			TextureFormat format;

		public:
			Cubemap();
			Cubemap(int width);
			Cubemap(int width, TextureFormat format);
			Cubemap(int width, int height, TextureFormat format);
			~Cubemap();

			std::vector<unsigned char> getRawCubemapData();
			std::vector<Color> getPixels(CubemapFace face);
			Color getPixel(CubemapFace face, int x, int y);
			
			void setRawCubemapData(std::vector<unsigned char> data);
			void setPixels(CubemapFace face, int x, int y, Color color);
			void setPixel(CubemapFace face, int x, int y, Color color);

			void readPixels();
			void apply();
	};
}

#endif































// #ifndef __CUBEMAP_H__
// #define __CUBEMAP_H__

// #include <vector>
// #include <string>

// #include "../core/Texture.h"
// #include "../core/Color.h"

// namespace PhysicsEngine
// {
// 	typedef enum CubemapFace
// 	{
// 		PositiveX,
// 		NegativeX,
// 		PositiveY,
// 		NegativeY,
// 		PositiveZ,
// 		NegativeZ
// 	};

// 	class Cubemap : public Texture
// 	{
// 		private:
// 			std::vector<unsigned char> rawCubemapData;

// 			TextureFormat format;
// 			GLuint handle;

// 		public:
// 			Cubemap();
// 			Cubemap(int width);
// 			Cubemap(int width, TextureFormat format);
// 			Cubemap(int width, int height, TextureFormat format);
// 			~Cubemap();

// 			void generate() override;
// 			void destroy() override;
// 			void bind() override;
// 			void unbind() override;

// 			void active(unsigned int slot);

// 			std::vector<unsigned char> getRawCubemapData();
// 			std::vector<Color> getPixels(CubemapFace face);
// 			Color getPixel(CubemapFace face, int x, int y);
			
// 			void setRawCubemapData(std::vector<unsigned char> data);
// 			void setPixels(CubemapFace face, int x, int y, Color color);
// 			void setPixel(CubemapFace face, int x, int y, Color color);

// 			void readPixels();
// 			void apply();

// 			GLuint getHandle() const;
// 	};
// }

// #endif