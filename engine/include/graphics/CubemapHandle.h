#ifndef CUBEMAP_HANDLE_H__
#define CUBEMAP_HANDLE_H__

#include <vector>

#include "../core/Texture.h"

namespace PhysicsEngine
{
class CubemapHandle
{
  protected:
    TextureFormat mFormat;
    TextureWrapMode mWrapMode;
    TextureFilterMode mFilterMode;
    int mWidth;

  public:
    CubemapHandle(int width, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode);
    virtual ~CubemapHandle() = 0;

    TextureFormat getFormat() const;
    TextureWrapMode getWrapMode() const;
    TextureFilterMode getFilterMode() const;
    int getWidth() const;

    virtual void load(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                      const std::vector<unsigned char> &data) = 0;
    virtual void update(TextureWrapMode wrapMode, TextureFilterMode filterMode) = 0;
    virtual void readPixels(std::vector<unsigned char> &data) = 0;
    virtual void writePixels(const std::vector<unsigned char> &data) = 0;
    virtual void bind(unsigned int texUnit) = 0;
    virtual void unbind(unsigned int texUnit) = 0;

    virtual void *getHandle() = 0;

    static CubemapHandle *create(int width, TextureFormat format, TextureWrapMode wrapMode,
                                 TextureFilterMode filterMode);
};
} // namespace PhysicsEngine

#endif