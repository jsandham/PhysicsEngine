#ifndef OPENGL_RENDER_TEXTURE_HANDLE_H__
#define OPENGL_RENDER_TEXTURE_HANDLE_H__

#include "../../RenderTextureHandle.h"

namespace PhysicsEngine
{
	class OpenGLRenderTextureHandle : public RenderTextureHandle
	{
	private:
		unsigned int mHandle;

	public:
        OpenGLRenderTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                  TextureFilterMode filterMode);
        ~OpenGLRenderTextureHandle();
		
		void* getTexture() override;
		void* getIMGUITexture() override;
	};
}

#endif