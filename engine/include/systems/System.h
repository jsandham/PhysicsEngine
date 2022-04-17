#ifndef SYSTEM_H__
#define SYSTEM_H__

#include "../core/Input.h"
#include "../core/Object.h"
#include "../core/Time.h"
#include "../core/Types.h"

namespace PhysicsEngine
{
class System : public Object
{
  protected:
    size_t mOrder;

    World *mWorld;

  public:
    bool mEnabled;

  public:
    System(World* world);
    System(World* world, Id id);
    ~System();

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