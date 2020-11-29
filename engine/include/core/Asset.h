#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Guid.h"
#include "Types.h"

namespace PhysicsEngine
{
class World;

class Asset
{
  protected:
    Guid mAssetId;
    std::string mAssetName;

  public:
    Asset();
    virtual ~Asset() = 0;

    virtual std::vector<char> serialize() const = 0;
    virtual std::vector<char> serialize(Guid assetId) const = 0;
    virtual void deserialize(const std::vector<char> &data) = 0;

    Guid getId() const;
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
template <typename T> struct IsAsset
{
    static constexpr bool value = false;
};
template <typename T> struct IsAssetInternal
{
    static constexpr bool value = false;
};

template <> struct IsAsset<Asset>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif