#ifndef __POOL_ALLOCATOR_H__
#define __POOL_ALLOCATOR_H__

#include <iostream>
#include <vector>

namespace PhysicsEngine
{
	template <class T, size_t T_per_page = 200>
	class PoolAllocator
	{
		private:
			const size_t pool_size = T_per_page * sizeof(T);
			std::vector<T *> pools;
			size_t count;
			size_t next_pos; //next_pos is always between 0 and T_per_page
			size_t end_pos;  //end_pos is always between 0 and T_per_page

			void alloc_pool() {
				std::cout << "alloc_pool called. Current size of pools: " << pools.size() << " reserving to havesize at least: " << pools.size() + 1 << std::endl;
				pools.reserve(pools.size() + 1);
				void *temp = operator new(pool_size);
				next_pos = 0;
				end_pos = 0;
				pools.push_back(static_cast<T *>(temp));

				std::cout << "pools size: " << pools.size() << std::endl;
			}

			void* allocate() {
				// if (next_pos == T_per_page) // might need to change this to if(count == getCapacity())
				if (count == getCapacity())
					alloc_pool();

				void* ret = pools.back() + next_pos;
				++next_pos;
				++end_pos;
				++count;
				return ret;
			}

		public:
			// PoolAllocator() : count(0), next_pos(T_per_page){}
			PoolAllocator() : count(0), next_pos(0), end_pos(0){}

			PoolAllocator(const PoolAllocator& other) = delete;

			PoolAllocator(PoolAllocator&& other)
			{
				std::cout << "Pool allocator move constructor called" << std::endl;
				pools = std::move(other.pools);
				count = other.count;
				next_pos = other.next_pos;
				end_pos = other.end_pos;
			}

			PoolAllocator& operator=(const PoolAllocator& other) = delete;

			PoolAllocator& operator=(PoolAllocator&& other)
			{
				std::cout << "Pool allocator move assignment operator called" << std::endl;
				this->~PoolAllocator();
				pools = std::move(other.pools);
				count = other.count;
				next_pos = other.next_pos;
				end_pos = other.end_pos;

				return *this;
			}

			T* construct() {
				return new(allocate()) T();
			}

			T* construct(std::vector<char> data) {
				return new(allocate()) T(data);
			}

			size_t getCount() const
			{
				return count;
			}

			size_t getCapacity() const
			{
				return T_per_page * pools.size();
			}

			T* get(size_t index) const
			{
				if (index < 0 || index >= getCount()) { return NULL; }

				size_t poolIndex = index / T_per_page;
				return pools[poolIndex] + (index % T_per_page);
			}

			T* getLast() const
			{
				return get(getCount() - 1);
			}

			
			T* destruct(size_t index) // this assumes I make the change above in allocate to use count == getCapacity()
			{
				//std::cout << "PoolAllocator destruct count: " << count << " next_pos: " << next_pos << " pools size: " << pools.size() << " index: " << index << std::endl;
				if (index < 0 || index >= getCount()) { return NULL; }

				T* current = NULL;
				if(index < count - 1){
					current = get(index);
					T* last = get(count - 1);
					//T* last = pools[pools.size() - 1] + (next_pos - 1);

					*current = *last;
				}

				//next_pos--; //next_pos is always between 0 and T_per_page
				next_pos = ((next_pos - 1) + T_per_page) % T_per_page;
				count--;

				return current;
			}

			~PoolAllocator() {
				std::cout << "POOL ALLOCATOR DESTRUCTOR CALLED" << std::endl;

				size_t originalSize = pools.size();
				while (!pools.empty()) {
					T *p = pools.back();
					size_t start = T_per_page;
					if (pools.size() == originalSize){
						start = end_pos;
					}

					std::cout << "start: " << start << std::endl;
					for (size_t pos = start; pos > 0; --pos)
					{
						std::cout << "pos: " << pos << std::endl;
						p[pos - 1].~T();
					}
					operator delete(static_cast<void *>(p));
					pools.pop_back();
				}
			}
	};

	template<class T>
	PoolAllocator<T>& getAllocator()
	{
		static PoolAllocator<T> allocator;

		return allocator;
	}

}

#endif