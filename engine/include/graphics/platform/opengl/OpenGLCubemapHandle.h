#ifndef OPENGL_CUBEMAP_HANDLE_H__
#define OPENGL_CUBEMAP_HANDLE_H__

#include "../../CubemapHandle.h"

namespace PhysicsEngine
{
class OpenGLCubemapHandle : public CubemapHandle
{
  private:
    unsigned int mHandle;

  public:
    OpenGLCubemapHandle(int width, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode);
    ~OpenGLCubemapHandle();

    void load(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
              const std::vector<unsigned char> &data) override;
    void update(TextureWrapMode wrapMode, TextureFilterMode filterMode) override;
    void readPixels(std::vector<unsigned char> &data) override;
    void writePixels(const std::vector<unsigned char> &data) override;
    void bind(unsigned int texUnit) override;
    void unbind(unsigned int texUnit) override;
    void *getHandle() override;
};
} // namespace PhysicsEngine

#endif
