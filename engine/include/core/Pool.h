#ifndef __POOL_H__
#define __POOL_H__

namespace PhysicsEngine
{
	template<class T>
	class Pool
	{
		private:
			int index;
			int size;
			T* array;

		public:
			Pool(int size = 200) : size(size)
			{
				index = -1;
				array = new T[size];
			}

			~Pool()
			{
				delete [] array;
			}

			int getIndex()
			{
				return index;
			}

			T* get(int index)
			{
				if(index >= 0 && index < size){
					return &array[index];
				}

				return NULL;
			}

			void allocate()
			{
				if(index < size - 1)
				{
					index++;
				}
			}

			void deallocate()
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