//#ifndef __TEXTURE3D_H__
//#define __TEXTURE3D_H__
//
//#include <vector>
//
//#include "Texture.h"
//#include "Color.h"
//
//namespace PhysicsEngine
//{
//	class Texture3D : public Texture
//	{
//		private:
//			std::vector<unsigned char> rawTextureData;
//
//		public:
//			Texture3D();
//			Texture3D(int width, int height, int depth, int numChannels);
//			~Texture3D();
//
//			void generate() override;
//			void destroy() override;
//			void bind() override;
//			void unbind() override;
//
//			void active(unsigned int slot);
//
//			std::vector<unsigned char> getRawTextureData();
//			Color getPixel(int x, int y);
//
//			void setRawTextureData(std::vector<unsigned char> data);
//			void setPixel(int x, int y, Color color);
//
//			GLuint getHandle() const;
//	};
//}
//
//#endif