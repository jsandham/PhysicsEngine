#ifndef ASSET_H__
#define ASSET_H__

#include <string>

#include "Object.h"
#include "Types.h"

namespace PhysicsEngine
{
class World;

class Asset : public Object
{
  protected:
    std::string mName;

  public:
    Asset(World *world);
    Asset(World *world, const Guid& id);
    ~Asset();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    void writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    std::string getName() const;
    void setName(const std::string &name);

    static bool isInternal(int type);

  private:
    friend class World;
};

template <typename T> struct AssetType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};
template <typename T> struct IsAssetInternal
{
    static constexpr bool value = false;
};
} // namespace PhysicsEngine

#endif