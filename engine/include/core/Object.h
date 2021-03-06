#ifndef OBJECT_H__
#define OBJECT_H__

#include <istream>
#include <ostream>
#include <vector>

#include "Guid.h"

//#include "../../../yaml-cpp/include/yaml-cpp/yaml.h"
#include "YamlExtensions.h"
#include "yaml-cpp/yaml.h"

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
    virtual void serialize(YAML::Node& out) const;
    virtual void deserialize(const YAML::Node& in);

    virtual int getType() const = 0;
    virtual std::string getObjectName() const = 0;

    Guid getId() const;
};

} // namespace PhysicsEngine

#endif