#ifndef SCENE_H__
#define SCENE_H__

#include <string>

#include "Object.h"

namespace PhysicsEngine
{
class Scene : public Object
{
  private:
    std::string mName;

  public:
    Scene(World* world);
    Scene(World* world, Guid id);
    ~Scene();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void writeToYAML(const std::string &filepath) const;

    void load(const std::string &filepath);

    std::string getName() const;
};

template <typename T> struct SceneType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};

template <typename> struct IsSceneInternal
{
    static constexpr bool value = false;
};

template <> struct IsSceneInternal<Scene>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif