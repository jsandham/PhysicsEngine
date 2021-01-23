#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include "Guid.h"

namespace PhysicsEngine
{
template <class T> Guid ExtactAssetId(const std::vector<char> &data)
{
    static_assert(std::is_base_of<Asset,T>(), "'T' is not of type Asset");

    return Guid::INVALID;
}

template <class T> Guid ExtactComponentId(const std::vector<char> &data)
{
    static_assert(std::is_base_of<Component,T>(), "'T' is not of type Component");

    return Guid::INVALID;
}

template <class T> Guid ExtactSystemId(const std::vector<char> &data)
{
    static_assert(std::is_base_of<System,T>(), "'T' is not of type System");

    return Guid::INVALID;
}
} // namespace PhysicsEngine

#endif