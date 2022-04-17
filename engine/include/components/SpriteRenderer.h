#ifndef SPRITERENDERER_H__
#define SPRITERENDERER_H__

#include "Component.h"

#include "../core/Color.h"

namespace PhysicsEngine
{
class SpriteRenderer : public Component
{
  private:
    Id mSpriteId;

  public:
    Color mColor;
    bool mFlipX;
    bool mFlipY;
    bool mSpriteChanged;
    bool mIsStatic;
    bool mEnabled;

  public:
    SpriteRenderer(World *world);
    SpriteRenderer(World *world, Id id);
    ~SpriteRenderer();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void setSprite(Id id);
    Id getSprite() const;
};

template <> struct ComponentType<SpriteRenderer>
{
    static constexpr int type = PhysicsEngine::SPRITERENDERER_TYPE;
};

template <> struct IsComponentInternal<SpriteRenderer>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif
