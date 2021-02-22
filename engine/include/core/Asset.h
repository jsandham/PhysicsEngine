#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Guid.h"
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
    Asset();
    Asset(Guid id);
    ~Asset();

    virtual void serialize(std::ostream &out) const;
    virtual void deserialize(std::istream &in);

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