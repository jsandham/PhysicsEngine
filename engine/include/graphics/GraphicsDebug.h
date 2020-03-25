#ifndef __GRAPHICS_DEBUG_H__
#define __GRAPHICS_DEBUG_H__	

#include "../core/Shader.h"
#include "../core/InternalShaders.h"
#include "../core/Texture2D.h"

#include "Graphics.h"

namespace PhysicsEngine
{
	typedef struct GraphicsDebug
	{
		Framebuffer mFBO[4];
		Shader mShaders[4];

		void init()
		{
			mShaders[0].setVertexShader(InternalShaders::depthMapVertexShader);
			mShaders[0].setFragmentShader(InternalShaders::depthMapFragmentShader);
			mShaders[1].setVertexShader(InternalShaders::normalMapVertexShader);
			mShaders[1].setFragmentShader(InternalShaders::normalMapFragmentShader);
			mShaders[2].setVertexShader(InternalShaders::overdrawVertexShader);
			mShaders[2].setFragmentShader(InternalShaders::overdrawFragmentShader);
			mShaders[3].setVertexShader(InternalShaders::lineVertexShader);
			mShaders[3].setFragmentShader(InternalShaders::lineFragmentShader);

			mShaders[0].compile();
			mShaders[1].compile();
			mShaders[2].compile();
			mShaders[3].compile();

			for(int i = 0; i < 4; i++){
				mFBO[i].mColorBuffer.redefine(1000, 1000, TextureFormat::RGB);
				mFBO[i].mDepthBuffer.redefine(1000, 1000, TextureFormat::Depth);

				glGenFramebuffers(1, &(mFBO[i].mHandle));
				glBindFramebuffer(GL_FRAMEBUFFER, mFBO[i].mHandle);

				// color
				mFBO[i].mColorBuffer.create();

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mFBO[i].mColorBuffer.getNativeGraphics(), 0);

				// depth
				mFBO[i].mDepthBuffer.create();

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mFBO[i].mDepthBuffer.getNativeGraphics(), 0);

				std::cout << "frame buffer handle: " << mFBO[i].mHandle << " framebuffer buffer handle: " << mFBO[i].mColorBuffer.getNativeGraphics() << " framebuffer depth buffer handle: " << mFBO[i].mDepthBuffer.getNativeGraphics() << std::endl;

				GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
				glDrawBuffers(1, DrawBuffers);

				if ((mFBO[i].mFramebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
					std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << mFBO[i].mFramebufferStatus << std::endl;
				}
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
		}
	}GraphicsDebug;
}


#endif