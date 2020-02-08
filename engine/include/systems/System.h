#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <vector>

#include "../core/Input.h"
#include "../core/Guid.h"

namespace PhysicsEngine
{
	class World;

	class System
	{
		protected:
			int order;

			World* world;

		public:
			Guid systemId;

		public:
			System();
			virtual ~System() = 0;

			virtual std::vector<char> serialize() = 0;
			virtual void deserialize(std::vector<char> data) = 0;

			virtual void init(World* world) = 0;
			virtual void update(Input input) = 0;

			int getOrder() const;
	};

	template <typename T>
	struct SystemType { static const int type; };

	template <typename T>
	const int SystemType<T>::type = -1;

	template< typename T>
	struct IsSystem { static bool value; };

	template<typename T>
	bool IsSystem<T>::value = false;

	template<>
	bool IsSystem<System>::value = true;
}

#endif