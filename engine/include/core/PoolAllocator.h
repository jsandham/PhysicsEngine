#ifndef POOL_ALLOCATOR_H__
#define POOL_ALLOCATOR_H__

#define NOMINMAX

#include <algorithm>
#include <iostream>
#include <vector>

#include "Allocator.h"
#include "Guid.h"
#include "Id.h"

#include "yaml-cpp/yaml.h"

namespace PhysicsEngine
{
class World;

template <class T, size_t T_per_page = 512> class PoolAllocator : public Allocator
{
  private:
    const size_t pool_size = T_per_page * sizeof(T);
    std::vector<T *> pools;
    size_t count;

    void alloc_pool()
    {
        pools.reserve(pools.size() + 1);
        void *temp = operator new(pool_size);
        pools.push_back(static_cast<T *>(temp));
    }

    void *allocate()
    {
        if (count == getCapacity())
            alloc_pool();

        size_t poolIndex = count / T_per_page;
        void *ret = pools[poolIndex] + (count % T_per_page);

        ++count;
        return ret;
    }

  public:
    PoolAllocator() : count(0)
    {
    }

    PoolAllocator(const PoolAllocator &other) = delete;

    PoolAllocator(PoolAllocator &&other)
    {
        pools = std::move(other.pools);
        count = other.count;
    }

    PoolAllocator &operator=(const PoolAllocator &other) = delete;

    PoolAllocator &operator=(PoolAllocator &&other)
    {
        this->~PoolAllocator();
        pools = std::move(other.pools);
        count = other.count;

        return *this;
    }

    T *construct(World *world, const Guid &guid, const Id &id)
    {
        return new (allocate()) T(world, guid, id);
    }

    T *construct(World *world, const YAML::Node &in, const Id &id)
    {
        T *t = new (allocate()) T(world, id);
        t->deserialize(in);
        return t;
    }

    size_t getCount() const override
    {
        return count;
    }

    size_t getCapacity() const override
    {
        return T_per_page * pools.size();
    }

    T *get(size_t index) const
    {
        if (index >= count)
        {
            return nullptr;
        }

        size_t poolIndex = index / T_per_page;
        return pools[poolIndex] + (index % T_per_page);
    }

    T *getLast() const
    {
        return (count > 0) ? get(count - 1) : nullptr;
    }

    T *destruct(size_t index)
    {
        if (index >= count)
        {
            return nullptr;
        }

        T *current = nullptr;
        T *last = get(count - 1);
        if (index < count - 1)
        {
            current = get(index);

            *current =
                std::move(*last); // as long as assignment operator ("rule of three") is implemented this will work
        }

        (*last).~T();

        count--;

        return current;
    }

    ~PoolAllocator()
    {
        for (size_t i = 0; i < pools.size(); i++)
        {
            T *p = pools[i];

            size_t start = (std::min)(count, T_per_page);

            for (size_t pos = 0; pos < start; pos++)
            {
                p[pos].~T();
            }

            operator delete(static_cast<void *>(p));

            count -= start;
        }
    }
};
} // namespace PhysicsEngine

#endif