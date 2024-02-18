#ifndef LIGHT_H__
#define LIGHT_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"

#include "../graphics/Framebuffer.h"
#include "../graphics/RenderTextureHandle.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
struct LightTargets
{
    Framebuffer *mShadowCascadeFBO[5];
    Framebuffer *mShadowSpotlightFBO;
    Framebuffer *mShadowCubemapFBO;
};

class World;

class Light
{
  public:
    HideFlag mHide;

    glm::vec4 mColor;
    float mIntensity;
    float mSpotAngle;
    float mInnerSpotAngle;
    float mShadowStrength;
    float mShadowNearPlane;
    float mShadowFarPlane;
    float mShadowBias;
    LightType mLightType;
    ShadowType mShadowType;
    bool mEnabled;

  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

    ShadowMapResolution mShadowMapResolution;
    LightTargets mTargets;

  public:
    Light(World *world, const Id &id);
    Light(World *world, const Guid &guid, const Id &id);
    ~Light();
    Light(const Light &other) = delete;
    Light &operator=(const Light &other) = delete;
    Light(Light &&other) = delete;
    Light &operator=(Light &&other);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    void resizeTargets();

    bool isShadowMapResolutionChanged() const;
    void setShadowMapResolution(ShadowMapResolution resolution);
    ShadowMapResolution getShadowMapResolution() const;

    glm::mat4 getProjMatrix() const;

    Framebuffer *getNativeGraphicsShadowCascadeFBO(int index) const;
    Framebuffer *getNativeGraphicsShadowSpotlightFBO() const;
    Framebuffer *getNativeGraphicsShadowCubemapFBO() const;

    RenderTextureHandle *getNativeGraphicsShadowCascadeDepthTex(int index) const;
    RenderTextureHandle *getNativeGrpahicsShadowSpotlightDepthTex() const;
    RenderTextureHandle *getNativeGraphicsShadowCubemapDepthTex() const;

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif