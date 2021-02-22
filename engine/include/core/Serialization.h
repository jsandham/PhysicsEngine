#ifndef __SERIALIZATION_H__
#define __SERIALIZATION_H__

#include <algorithm>
#include <istream>
#include <ostream>
#include <string>

#include "Guid.h"

namespace PhysicsEngine
{
// Write

template <class T> void write(std::ostream &out, T value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <class T> void write(std::ostream &out, T *ptr, size_t size)
{
    out.write(reinterpret_cast<const char *>(ptr), size * sizeof(T));
}

// Write template specializations

template <> inline void PhysicsEngine::write(std::ostream &out, bool value)
{
    uint8_t temp = static_cast<uint8_t>(value);

    out.write(reinterpret_cast<const char *>(&temp), sizeof(uint8_t));
}

template <> inline void PhysicsEngine::write(std::ostream &out, size_t value)
{
    uint64_t temp = static_cast<uint64_t>(value);

    out.write(reinterpret_cast<const char *>(&temp), sizeof(uint64_t));
}

template <> inline void PhysicsEngine::write(std::ostream &out, int value)
{
    int32_t temp = static_cast<int32_t>(value);

    out.write(reinterpret_cast<const char *>(&temp), sizeof(int32_t));
}

template <> inline void PhysicsEngine::write(std::ostream &out, std::string value)
{
    char temp[64];
    std::size_t len = std::min(size_t(64 - 1), value.size());
    memcpy(&temp[0], &value[0], len);
    temp[len] = '\0';

    out.write(reinterpret_cast<const char *>(&temp), 64 * sizeof(char));
}

// Read

template <class T> void read(std::istream &in, T &value)
{
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
}

template <class T> void read(std::istream &in, T *ptr, size_t size)
{
    in.read(reinterpret_cast<char *>(ptr), size * sizeof(T));
}

// Read template specializations

template <> inline void PhysicsEngine::read(std::istream &in, bool &value)
{
    uint8_t temp;
    in.read(reinterpret_cast<char *>(&temp), sizeof(uint8_t));
    value = static_cast<bool>(temp);
}

template <> inline void PhysicsEngine::read(std::istream &in, size_t &value)
{
    uint64_t temp;
    in.read(reinterpret_cast<char *>(&temp), sizeof(uint64_t));
    value = static_cast<size_t>(temp);
}

template <> inline void PhysicsEngine::read(std::istream &in, int &value)
{
    int32_t temp;
    in.read(reinterpret_cast<char *>(&temp), sizeof(int32_t));
    value = static_cast<int>(temp);
}

template <> inline void PhysicsEngine::read(std::istream &in, std::string &value)
{
    char temp[64];

    in.read(reinterpret_cast<char *>(&temp), 64 * sizeof(char));

    value = std::string(temp);
}
} // namespace PhysicsEngine

#endif