#ifndef __ASSET_H__
#define __ASSET_H__

#include <string>

#include "Guid.h"

namespace PhysicsEngine
{
	class Manager;

	class Asset
	{
		protected:
			Manager* manager;

		public:
			Guid assetId;

		public:
			Asset();
			virtual ~Asset() = 0;

			void setManager(Manager* manager);

			template <typename T>
			static int getInstanceType()
			{
				// static variables only run the first time the function is called
			    static int id = nextValue();
			    return id;
			}

		private:
			static int nextValue()
			{
				// static variables only run the first time the function is called
			    static int id = 0;
			    int result = id;
			    ++id;
			    return result;
			}

	};
}

#endif