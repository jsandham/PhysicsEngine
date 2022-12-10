#ifndef OPENGL_VERTEX_BUFFER_H__
#define OPENGL_VERTEX_BUFFER_H__

#include "../../VertexBuffer.h"

namespace PhysicsEngine
{
	class OpenGLVertexBuffer : public VertexBuffer
	{
	public:
		size_t mSize;
		unsigned int mBuffer;

	public:
		OpenGLVertexBuffer();
		~OpenGLVertexBuffer();

		void resize(size_t size) override;
		//void setData(const void* data, size_t size) override;
		void bind() override;
		void unbind() override;
		void* get() override;
	};
}

#endif