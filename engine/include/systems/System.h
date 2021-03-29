#ifndef SYSTEM_H__
#define SYSTEM_H__

#include <vector>

#include "../core/Guid.h"
#include "../core/Input.h"
#include "../core/Object.h"
#include "../core/Time.h"
#include "../core/Types.h"

namespace PhysicsEngine
{
class World;

class System : public Object
{
  protected:
    size_t mOrder;

    World *mWorld;

  public:
    System();
    System(Guid id);
    ~System();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual void init(World *world) = 0;
    virtual void update(const Input &input, const Time &time) = 0;

    size_t getOrder() const;

    static bool isInternal(int type);

  private:
    friend class World;
};

template <typename T> struct SystemType
{
    static constexpr int type = PhysicsEngine::INVALID_TYPE;
};
template <typename T> struct IsSystemInternal
{
    static constexpr bool value = false;
};
} // namespace PhysicsEngine

#endif