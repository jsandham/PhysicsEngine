#ifndef __POOL_H__
#define __POOL_H__

#include<vector>

namespace PhysicsEngine
{
	template <class T>
	class Pool
	{
		private:
			unsigned int next;
			std::vector<T> pool;
		
		public:
			Pool(unsigned int size = 10) : pool(size), next(0){}
			~Pool(){}

			T* getNext()
			{
				if (next == pool.size()){
					std::cout << "Error: pool exausted" << std::endl;
					//pool.push_back(T());
				}

				return &pool[next++];
			}

			std::vector<T> getPool()
			{
				return pool;
			}

			void swapWithLast(unsigned int index)
			{
				T transform = pool[--next];
				pool[index] = transform;
				//pool[next].clear();
			}
	};
}

#endif