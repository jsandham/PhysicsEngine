#ifndef __GRAPHICS_DEBUG_H__
#define __GRAPHICS_DEBUG_H__	

#include "../core/Shader.h"
#include "../core/Texture2D.h"

#include "Graphics.h"

namespace PhysicsEngine
{
	struct GraphicsDebug
	{
		Framebuffer fbo[3];
		Shader shaders[3];

		void init()
		{
			shaders[0].vertexShader = Shader::depthMapVertexShader;
			shaders[0].fragmentShader = Shader::depthMapFragmentShader;
			shaders[1].vertexShader	= Shader::normalMapVertexShader;
			shaders[1].fragmentShader = Shader::normalMapFragmentShader;
			shaders[2].vertexShader	= Shader::overdrawVertexShader;
			shaders[2].fragmentShader = Shader::overdrawFragmentShader;

			shaders[0].compile();
			shaders[1].compile();
			shaders[2].compile();

			for(int i = 0; i < 3; i++){
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

				glGenTextures(1, &(fbo[i].colorBuffer.handle.handle));
				glBindTexture(GL_TEXTURE_2D, fbo[i].colorBuffer.handle.handle);

				GLenum openglFormat = Graphics::getTextureFormat(format);

				glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

				glBindTexture(GL_TEXTURE_2D, 0);

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo[i].colorBuffer.handle.handle, 0);

				// depth
				width = fbo[i].depthBuffer.getWidth();
				height = fbo[i].depthBuffer.getHeight();
				numChannels = fbo[i].depthBuffer.getNumChannels();
				format = fbo[i].depthBuffer.getFormat();
				rawTextureData = fbo[i].depthBuffer.getRawTextureData();

				glGenTextures(1, &(fbo[i].depthBuffer.handle.handle));
				glBindTexture(GL_TEXTURE_2D, fbo[i].depthBuffer.handle.handle);

				openglFormat = Graphics::getTextureFormat(format);

				glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

				glBindTexture(GL_TEXTURE_2D, 0);

				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fbo[i].depthBuffer.handle.handle, 0);

				std::cout << "frame buffer handle: " << fbo[i].handle << " framebuffer buffer handle: " << fbo[i].colorBuffer.handle.handle << " framebuffer depth buffer handle: " << fbo[i].depthBuffer.handle.handle << std::endl;

				GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
				glDrawBuffers(1, DrawBuffers);

				if ((fbo[i].framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
					std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << fbo[i].framebufferStatus << std::endl;
				}
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
			
		}
	};
}


#endif