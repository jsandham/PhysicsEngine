#ifndef INDEX_BUFFER_H__
#define INDEX_BUFFER_H__

namespace PhysicsEngine
{
	class IndexBuffer
	{
	protected:
		size_t mSize;
	public:
		IndexBuffer();
		IndexBuffer(const IndexBuffer& other) = delete;
		IndexBuffer& operator=(const IndexBuffer& other) = delete;
		virtual ~IndexBuffer() = 0;

		virtual void resize(size_t size) = 0;
		virtual void setData(void* data, size_t offset, size_t size) = 0;
		virtual void bind() = 0;
		virtual void unbind() = 0;
		virtual void* getBuffer() = 0;

		size_t getSize() const;

		static IndexBuffer* create();
	};
}

#endif