#ifndef __TEXTURE3D_H__
#define __TEXTURE3D_H__

#include <vector>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
	class Texture3D : public Texture
	{
		private:
			int width;
			int height;
			int depth;

		public:
			Texture3D();
			Texture3D(unsigned char* data);
			Texture3D(int width, int height, int depth, int numChannels);
			~Texture3D();

			int getWidth() const;
			int getHeight() const;
			int getDepth() const;

			void redefine(int width, int height, int depth, TextureFormat format);

			std::vector<unsigned char> getRawTextureData();
			Color getPixel(int x, int y, int z);
			TextureFormat getFormat();

			void setRawTextureData(std::vector<unsigned char> data);
			void setPixel(int x, int y, int z, Color color);

			void readPixels();
			void apply();
	};
}

#endif