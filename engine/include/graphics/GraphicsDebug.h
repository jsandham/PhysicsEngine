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
				int width = fbo[i].colorBuffer.getWidth();
				int height = fbo[i].colorBuffer.getHeight();
				int numChannels = fbo[i].colorBuffer.getNumChannels();
				TextureFormat format = fbo[i].colorBuffer.getFormat();
				std::vector<unsigned char> rawTextureData = fbo[i].colorBuffer.getRawTextureData();

				glGenTextures(1, &(fbo[i].colorBuffer.tex));
				glBindTexture(GL_TEXTURE_2D, fbo[i].colorBuffer.tex);

				GLenum openglFormat = Graphics::getTextureFormat(format);

				glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

				glBindTexture(GL_TEXTURE_2D, 0);

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo[i].colorBuffer.tex, 0);

				// depth
				width = fbo[i].depthBuffer.getWidth();
				height = fbo[i].depthBuffer.getHeight();
				numChannels = fbo[i].depthBuffer.getNumChannels();
				format = fbo[i].depthBuffer.getFormat();
				rawTextureData = fbo[i].depthBuffer.getRawTextureData();

				glGenTextures(1, &(fbo[i].depthBuffer.tex));
				glBindTexture(GL_TEXTURE_2D, fbo[i].depthBuffer.tex);

				openglFormat = Graphics::getTextureFormat(format);

				glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

				glBindTexture(GL_TEXTURE_2D, 0);

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fbo[i].depthBuffer.tex, 0);

				std::cout << "frame buffer handle: " << fbo[i].handle << " framebuffer buffer handle: " << fbo[i].colorBuffer.tex << " framebuffer depth buffer handle: " << fbo[i].depthBuffer.tex << std::endl;

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