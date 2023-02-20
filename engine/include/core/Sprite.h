#ifndef SPRITE_H__
#define SPRITE_H__

#include <string>
#include <vector>

#include "Asset.h"
#include "Guid.h"

#include "../graphics/VertexBuffer.h"
#include "../graphics/MeshHandle.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"

namespace PhysicsEngine
{
class Sprite : public Asset
{
  private:
    Guid mTextureId;
    MeshHandle *mHandle;
    VertexBuffer *mBuffer;
    bool mChanged;

  public:
    int mPixelsPerUnit;

  public:
    Sprite(World *world, const Id &id);
    Sprite(World *world, const Guid &guid, const Id &id);
    ~Sprite();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool isChanged() const;

    MeshHandle *getNativeGraphicsHandle() const;

    Guid getTextureId() const;
    void setTextureId(Guid textureId);
};

template <> struct AssetType<Sprite>
{
    static constexpr int type = PhysicsEngine::SPRITE_TYPE;
};

template <> struct IsAssetInternal<Sprite>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif