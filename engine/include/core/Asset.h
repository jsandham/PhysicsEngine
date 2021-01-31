#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Object.h"
#include "Guid.h"
#include "Types.h"

namespace PhysicsEngine
{
class World;

class Asset : public Object
{
  protected:
    std::string mAssetName;

  public:
    Asset();
    Asset(Guid id);
    virtual ~Asset() = 0;

    virtual std::vector<char> serialize() const = 0;
    virtual std::vector<char> serialize(Guid assetId) const = 0;
    virtual void deserialize(const std::vector<char>& data) = 0;

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

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