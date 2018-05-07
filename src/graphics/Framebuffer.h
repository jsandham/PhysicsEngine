#ifndef __FRAMEBUFFER_H__
#define __FRAMEBUFFER_H__

#include <GL/glew.h>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"


namespace PhysicsEngine
{
	class Framebuffer
	{
		private:
			int width;
			int height;

			GLenum framebufferStatus;
			GLuint handle;

		public:
			Framebuffer(int width, int height);
			~Framebuffer();

			void generate();
			void destroy();
			void bind();
			void unbind();

			void clearColorBuffer(glm::vec4 value);
			void clearDepthBuffer(float value);

			int getWidth() const;
			int getHeight() const;

			void addAttachment(GLuint textureHandle, GLenum attachmentType, GLint level);
			void addAttachment2D(GLuint textureHandle, GLenum attachmentType, GLenum textureTarget, GLint level);
	};
}

#endif