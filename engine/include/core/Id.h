#ifndef ID_H__
#define ID_H__

#include <functional>

namespace PhysicsEngine
{
    class Id
    {
    private:
        int mId;

    public:
        Id();
        Id(int id);
        ~Id();

        Id &operator=(const Id &id);
        bool operator==(const Id &id) const;
        bool operator!=(const Id &id) const;
        bool operator<(const Id &id) const;

        bool isValid() const;
        bool isInvalid() const;

        static Id newId();
        static const Id INVALID;
    };
}

// allow use of Guid in unordered_set and unordered_map
namespace std
{
template <> struct hash<PhysicsEngine::Id>
{
    size_t operator()(const PhysicsEngine::Id &id) const noexcept
    {
        static_assert(sizeof(PhysicsEngine::Id) == sizeof(int));

        const int *p = reinterpret_cast<const int *>(&id);
        std::hash<int> hash;
        return hash(*p);
    }
};
} // namespace std

#endif