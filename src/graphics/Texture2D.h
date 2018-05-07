#ifndef __TEXTURE2D_H__
#define __TEXTURE2D_H__

#include <string>
#include <vector>
#include <GL/glew.h>

#include "Texture.h"
#include "Color.h"

namespace PhysicsEngine
{
	class Texture2D : public Texture
	{
		private:
			std::vector<unsigned char> rawTextureData;

			TextureFormat format;
			GLuint handle;

		public:
			Texture2D();
			Texture2D(int width, int height);
			Texture2D(int width, int height, TextureFormat format);
			~Texture2D();

			void generate() override;
			void destroy() override;
			void bind() override;
			void unbind() override;

			void active(unsigned int slot);

			std::vector<unsigned char> getRawTextureData();
			std::vector<Color> getPixels();
			Color getPixel(int x, int y);

			void setRawTextureData(std::vector<unsigned char> data);
			void setPixels(std::vector<Color> colors);
			void setPixel(int x, int y, Color color);

			void readPixels();
			void apply();

			GLuint getHandle() const;
	};
}


#endif