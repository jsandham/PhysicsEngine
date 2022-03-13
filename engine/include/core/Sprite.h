#ifndef SPRITE_H__
#define SPRITE_H__

#include <string>
#include <vector>

#include "Asset.h"
#include "Guid.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"

namespace PhysicsEngine
{
class Sprite : public Asset
{
  private:
    Guid mTextureId;
    unsigned int mVao;
    // int mWidth;
    // int mHeight;
    bool mCreated;
    bool mChanged;

  public:
    int mPixelsPerUnit;

  public:
    Sprite(World *world);
    Sprite(World *world, const Guid& id);
    ~Sprite();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    bool isCreated() const;
    bool isChanged() const;

    unsigned int getNativeGraphicsVAO() const;

    Guid getTextureId() const;
    void setTextureId(Guid textureId);

    void create();
    void destroy();
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