#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <vector>

#include "../core/Object.h"
#include "../core/Guid.h"
#include "../core/Input.h"
#include "../core/Time.h"
#include "../core/Types.h"

namespace PhysicsEngine
{
class World;

class System : public Object
{
  protected:
    int mOrder;

    World *mWorld;

  public:
    System();
    System(Guid id);
    ~System();

    virtual std::vector<char> serialize() const = 0;
    virtual std::vector<char> serialize(const Guid& systemId) const = 0;
    virtual void deserialize(const std::vector<char>& data) = 0;

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    virtual void init(World *world) = 0;
    virtual void update(const Input &input, const Time &time) = 0;

    int getOrder() const;

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