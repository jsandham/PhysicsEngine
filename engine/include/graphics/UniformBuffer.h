#ifndef UNIFORM_BUFFER_H__
#define UNIFORM_BUFFER_H__

namespace PhysicsEngine
{
	class UniformBuffer
	{
	public:
		UniformBuffer(size_t size);
		virtual ~UniformBuffer() = 0;

		virtual void setData(void* data, size_t size, size_t offset, size_t bindingPoint) = 0;

		static UniformBuffer* create(size_t size);
	};
}

#endif
