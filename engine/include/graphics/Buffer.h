#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <iostream>
#include <GL/glew.h>
#include <gl/gl.h>

namespace PhysicsEngine
{
	class Buffer
	{
		private:
			GLenum target;
			GLenum usage;

		public:
			GLuint handle;

		public:
			Buffer();
			~Buffer();

			void generate(GLenum target, GLenum usage);
			void destroy();
			void bind();
			void unbind();

			void setData(const void* data, std::size_t size);
			void setSubData(const void* data, unsigned int offset, std::size_t size);
			void setRange(unsigned int index, unsigned int offset, std::size_t size);
	};
}

#endif