#ifndef UNIFORM_BUFFER_H__
#define UNIFORM_BUFFER_H__

namespace PhysicsEngine
{
	class UniformBuffer
	{
	public:
		UniformBuffer();
		UniformBuffer(const UniformBuffer &other) = delete;
        UniformBuffer &operator=(const UniformBuffer &other) = delete;
		virtual ~UniformBuffer() = 0;

		virtual size_t getSize() const = 0;
        virtual unsigned int getBindingPoint() const = 0;

		virtual void bind() = 0;
        virtual void unbind() = 0;
		virtual void setData(void* data, size_t offset, size_t size) = 0;

		static UniformBuffer* create(size_t size, unsigned int bindingPoint);
	};
}

#endif
