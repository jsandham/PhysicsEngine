#ifndef POOL_ALLOCATOR_H__
#define POOL_ALLOCATOR_H__

#define NOMINMAX

#include <algorithm>
#include <iostream>
#include <vector>

#include "Allocator.h"
#include "Guid.h"
#include "Id.h"

namespace PhysicsEngine
{
class World;

template <class T, size_t T_per_page = 512> class PoolAllocator //: public Allocator
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

    T *construct(World *world, const Id &id)
    {
        T *t = new (allocate()) T(world, id);
        return t;
    }

    T* construct()
    {
        T *t = new (allocate()) T();
        return t;
    }

    size_t getCount() const //override
    {
        return count;
    }

    size_t getCapacity() const //override
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








template <class T> struct myHash
{
};

template <> struct myHash<int>
{
    static size_t hashFunction(const int m)
    {
        return std::hash<int>()(m);
    }
};

//template <class K, class V, class HashGenerator = myHash<K>>

template <typename KEY_TYPE, typename VAL_TYPE, typename HashGenerator = myHash<KEY_TYPE>, size_t SIZE = 65536>
class hash_map
{
  private:
    std::array<KEY_TYPE, SIZE> keys;
    std::array<VAL_TYPE, SIZE> values;
    std::array<bool, SIZE> occupied;

  public:
    void insert(KEY_TYPE key, VAL_TYPE val)
    {
        size_t index = hash(key);
        if (!occupied[index] || keys[index] == key)
        {
            keys[index] = key;
            values[index] = val;
            occupied[index] = true;
            return;
        }

        for (size_t i = index + 1; keys.size(); i++)
        {
            if (!occupied[i] || keys[i] == key)
            {
                keys[i] = key;
                values[i] = val;
                occupied[i] = true;
                return;
            }           
        }

        for (size_t i = 0; index; i++)
        {
            if (!occupied[i] || keys[i] == key)
            {
                keys[i] = key;
                values[i] = val;
                occupied[i] = true;
                return;
            }
        }
    }

    void erase(KEY_TYPE key)
    {
        size_t index = hash(key);

        if (occupied[index] && keys[index] == key)
        {
            occupied[index] = false;
            return;
        }

        for (size_t i = index + 1; keys.size(); i++)
        {
            if (occupied[i] && keys[i] == key)
            {
                occupied[i] = false;
                return;
            }
        }

        for (size_t i = 0; index; i++)
        {
            if (occupied[i] && keys[i] == key)
            {
                occupied[i] = false;
                return;
            }
        }   
    }

    bool contains(KEY_TYPE key)
    {
        size_t index = hash(key);

        if (occupied[index] && keys[index] == key)
        {
            return true;
        }

        for (size_t i = index + 1; keys.size(); i++)
        {
            if (occupied[i] && keys[i] == key)
            {
                return true;
            }
        }

        for (size_t i = 0; index; i++)
        {
            if (occupied[i] && keys[i] == key)
            {
                return true;
            }
        }  

        return false;
    }

  private:
    size_t hash(const KEY_TYPE &k)
    {
        return HashGenerator::hashFunction(k) % keys.size();
    }
};

} // namespace PhysicsEngine

#endif