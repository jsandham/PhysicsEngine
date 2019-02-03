#ifndef __POOL_ALLOCATOR_H__
#define __POOL_ALLOCATOR_H__

namespace PhysicsEngine
{
	template <class T, size_t T_per_page = 200>
	class PoolAllocator 
	{
		private:
			const size_t pool_size = T_per_page * sizeof(T);
			std::vector<T *> pools;
			size_t next_pos;

			void alloc_pool() {
				next_pos = 0;
				void *temp = operator new(pool_size);
				pools.push_back(static_cast<T *>(temp));
			}

		public:
			PoolAllocator() {
				alloc_pool();
			}

			T* allocate() {
				if (next_pos == T_per_page)
					alloc_pool();

				T *ret = new(pools.back() + next_pos) T;
				++next_pos;
				return ret;
			}

			size_t getCount() const
			{
				return T_per_page * (pools.size() - 1) + next_pos;
			}

			T* get(size_t index) const
			{
				if (index >= count()) { return NULL; }

				size_t poolIndex = index / T_per_page;
				return pools[poolIndex] + (index % T_per_page);
			}

			~PoolAllocator() {
				while (!pools.empty()) {
					T *p = pools.back();
					for (size_t pos = T_per_page; pos > 0; --pos)
						p[pos - 1].~T();
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