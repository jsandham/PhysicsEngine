#ifndef __POOL_H__
#define __POOL_H__

#include <vector>

namespace PhysicsEngine
{
	template<class T>
	class Pool
	{
		private:
			int index;
			int size;
			std::vector<T> data;

		public:
			Pool(int size = 200) : size(size)
			{
				index = 0;
				data.resize(size);
			}

			~Pool()
			{
				
			}

			int getIndex()
			{
				return index;
			}

			T* get(int index)
			{
				if(index >= 0 && index < size){
					return &data[index];
				}

				return NULL;
			}

			void increment()
			{
				if(index < size - 1)
				{
					index++;
				}
			}

			void decrement()
			{
				if(index > 0){
					index--;
				}
			}

			void setIndex(int index)
			{
				this->index = index;
			}

	};
}

#endif