#ifndef __OBJECT_H__
#define __OBJECT_H__

#include <istream>
#include <ostream>
#include <vector>

#include "Guid.h"

//#include "../../../yaml-cpp/include/yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class Object
{
  protected:
    Guid mId;

  public:
    Object();
    Object(Guid id);
    virtual ~Object() = 0;

    virtual void serialize(std::ostream &out) const;
    virtual void deserialize(std::istream &in);

    Guid getId() const;
};

} // namespace PhysicsEngine

#endif