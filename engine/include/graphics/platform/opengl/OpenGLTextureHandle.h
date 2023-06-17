#ifndef OPENGL_TEXTURE_HANDLE_H__
#define OPENGL_TEXTURE_HANDLE_H__

#include "../../TextureHandle.h"

namespace PhysicsEngine
{
	class OpenGLTextureHandle : public TextureHandle
	{
	private:
		unsigned int mHandle;

	public:
		OpenGLTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                          TextureFilterMode filterMode);
		~OpenGLTextureHandle();

		void load(TextureFormat format,
			TextureWrapMode wrapMode,
			TextureFilterMode filterMode,
			int width,
			int height,
			const std::vector<unsigned char>& data) override;
        void update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel) override;
		void readPixels(std::vector<unsigned char>& data) override;
		void writePixels(const std::vector<unsigned char>& data) override;
		void bind(unsigned int texUnit) override;
		void unbind(unsigned int texUnit) override;
        void *getTexture() override;
        void *getIMGUITexture() override;
	};
}

#endif