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
		Framebuffer fbo[4];
		Shader shaders[4];

		void init()
		{
			shaders[0].setVertexShader(InternalShaders::depthMapVertexShader);
			shaders[0].setFragmentShader(InternalShaders::depthMapFragmentShader);
			shaders[1].setVertexShader(InternalShaders::normalMapVertexShader);
			shaders[1].setFragmentShader(InternalShaders::normalMapFragmentShader);
			shaders[2].setVertexShader(InternalShaders::overdrawVertexShader);
			shaders[2].setFragmentShader(InternalShaders::overdrawFragmentShader);
			shaders[3].setVertexShader(InternalShaders::lineVertexShader);
			shaders[3].setFragmentShader(InternalShaders::lineFragmentShader);

			shaders[0].compile();
			shaders[1].compile();
			shaders[2].compile();
			shaders[3].compile();

			for(int i = 0; i < 4; i++){
				fbo[i].colorBuffer.redefine(1000, 1000, TextureFormat::RGB);
				fbo[i].depthBuffer.redefine(1000, 1000, TextureFormat::Depth);

				glGenFramebuffers(1, &(fbo[i].handle));
				glBindFramebuffer(GL_FRAMEBUFFER, fbo[i].handle);

				// color
				fbo[i].colorBuffer.create();

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo[i].colorBuffer.getNativeGraphics(), 0);

				// depth
				fbo[i].depthBuffer.create();

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fbo[i].depthBuffer.getNativeGraphics(), 0);

				std::cout << "frame buffer handle: " << fbo[i].handle << " framebuffer buffer handle: " << fbo[i].colorBuffer.getNativeGraphics() << " framebuffer depth buffer handle: " << fbo[i].depthBuffer.getNativeGraphics() << std::endl;

				GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
				glDrawBuffers(1, DrawBuffers);

				if ((fbo[i].framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
					std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << fbo[i].framebufferStatus << std::endl;
				}
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
		}
	}GraphicsDebug;
}


#endif