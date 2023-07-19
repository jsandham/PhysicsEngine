#ifndef UNIFORM_BUFFER_H__
#define UNIFORM_BUFFER_H__

enum class PipelineStage
{
    VS,
    PS,
    GS
};

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

    virtual void bind(PipelineStage stage) = 0;
    virtual void unbind(PipelineStage stage) = 0;
    virtual void setData(const void *data, size_t offset, size_t size) = 0;
    virtual void getData(void *data, size_t offset, size_t size) = 0;
    virtual void copyDataToDevice() = 0;

    static UniformBuffer *create(size_t size, unsigned int bindingPoint);
};
} // namespace PhysicsEngine

#endif
