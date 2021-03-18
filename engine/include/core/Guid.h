#ifndef GUID_H__
#define GUID_H__

#include <ostream>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
// see https://github.com/graeme-hill/crossguid
class Guid
{
  private:
    unsigned char bytes[16];

  public:
    Guid();
    Guid(const std::vector<unsigned char> &bytes);
    Guid(const unsigned char *bytes);
    Guid(const Guid &guid);
    Guid(const std::string &str);
    ~Guid();

    Guid &operator=(const Guid &guid);
    bool operator==(const Guid &guid) const;
    bool operator!=(const Guid &guid) const;
    bool operator<(const Guid &guid) const;

    bool isEmpty() const;
    bool isValid() const;
    bool isInvalid() const;
    std::string toString() const;

    static Guid newGuid();

    static const Guid INVALID;
};

std::ostream &operator<<(std::ostream &os, const Guid &id);
} // namespace PhysicsEngine

namespace YAML
{
    // Guid
    template<>
    struct convert<PhysicsEngine::Guid> {
        static Node encode(const PhysicsEngine::Guid& rhs) {
            Node node;
            node = rhs.toString();
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Guid& rhs) {
            rhs = node.as<std::string>();
            return true;
        }
    };
}

// allow use of Guid in unordered_set and unordered_map
namespace std
{
template <> struct hash<PhysicsEngine::Guid>
{
    size_t operator()(const PhysicsEngine::Guid &guid) const noexcept
    {
        const std::uint64_t *p = reinterpret_cast<const std::uint64_t *>(&guid);
        std::hash<std::uint64_t> hash;
        return hash(p[0]) ^ hash(p[1]);
    }
};
} // namespace std

#endif