#ifndef __UNIFORMBUFFEROBJECT_H__
#define __UNIFORMBUFFEROBJECT_H__

#include <GL/glew.h>

namespace PhysicsEngine
{
	class UniformBufferObject
	{
		private:
			GLuint handle;

		public:
			UniformBufferObject();
			~UniformBufferObject();

			void generate();
			void destroy();
			void bind();
			void unbind();
			void setData(const void* data, std::size_t size);
			void setSubData(const void* data, unsigned int offset, std::size_t size);
			void setRange(unsigned int offset, std::size_t size);
	};
}

#endif