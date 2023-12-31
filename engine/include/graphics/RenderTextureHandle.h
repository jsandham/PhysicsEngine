#ifndef RENDER_TEXTURE_HANDLE_H__
#define RENDER_TEXTURE_HANDLE_H__

#include <vector>

#include "../core/AssetEnums.h"

namespace PhysicsEngine
{
class RenderTextureHandle
{
  protected:
    TextureFormat mFormat;
    TextureWrapMode mWrapMode;
    TextureFilterMode mFilterMode;
    int mAnisoLevel;
    int mWidth;
    int mHeight;

  public:
    RenderTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                        TextureFilterMode filterMode);
    RenderTextureHandle(const RenderTextureHandle &other) = delete;
    RenderTextureHandle &operator=(const RenderTextureHandle &other) = delete;
    virtual ~RenderTextureHandle() = 0;

    int getWidth() const;
    int getHeight() const;

    virtual void load(const std::vector<unsigned char> &data) = 0;

    virtual void *getTexture() = 0;
    virtual void *getIMGUITexture() = 0;

    static RenderTextureHandle *create(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                       TextureFilterMode filterMode);
};
} // namespace PhysicsEngine

#endif