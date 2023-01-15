#ifndef VERTEX_BUFFER_H__
#define VERTEX_BUFFER_H__

namespace PhysicsEngine
{
	class VertexBuffer
	{
    protected:
        size_t mSize;
	public:
		VertexBuffer();
		VertexBuffer(const VertexBuffer &other) = delete;
        VertexBuffer &operator=(const VertexBuffer &other) = delete;
		virtual ~VertexBuffer() = 0;

		virtual void resize(size_t size) = 0;
        virtual void setData(void* data, size_t offset, size_t size) = 0;
		virtual void bind() = 0;
		virtual void unbind() = 0;
		virtual void* getBuffer() = 0;

		size_t getSize() const;

        static VertexBuffer *create();
	};
}

#endif