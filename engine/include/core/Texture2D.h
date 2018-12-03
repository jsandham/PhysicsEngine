#ifndef __TEXTURE2D_H__
#define __TEXTURE2D_H__

#include <vector>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
	class Texture2D : public Texture
	{
		private:
			int width;
			int height;

		public:
			Texture2D();
			Texture2D(int width, int height);
			Texture2D(int width, int height, TextureFormat format);
			~Texture2D();

			int getWidth() const;
			int getHeight() const;

			void redefine(int width, int height, TextureFormat format);

			std::vector<unsigned char> getRawTextureData();
			std::vector<Color> getPixels();
			Color getPixel(int x, int y);
			TextureFormat getFormat();

			void setRawTextureData(std::vector<unsigned char> data, int width, int height, TextureFormat format);
			void setPixels(std::vector<Color> colors);
			void setPixel(int x, int y, Color color);

			void readPixels();
			void apply();
	};
}

#endif