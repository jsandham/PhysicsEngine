#ifndef __OBJECT_H__
#define __OBJECT_H__

#include <ostream>
#include <istream>
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

        virtual std::vector<char> serialize() const = 0;
        virtual void deserialize(const std::vector<char>& data) = 0;

        Guid getId() const;
    };

} // namespace PhysicsEngine

#endif