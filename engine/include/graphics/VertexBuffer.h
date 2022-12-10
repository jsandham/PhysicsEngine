#ifndef VERTEX_BUFFER_H__
#define VERTEX_BUFFER_H__

namespace PhysicsEngine
{
	class VertexBuffer
	{
	public:
		VertexBuffer();
		virtual ~VertexBuffer() = 0;

		virtual void resize(size_t size) = 0;
		virtual void bind() = 0;
		virtual void unbind() = 0;
		virtual void* get() = 0;

		static VertexBuffer* create();
	};
}

#endif