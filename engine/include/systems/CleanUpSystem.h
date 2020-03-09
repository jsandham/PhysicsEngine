#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

namespace PhysicsEngine
{
	class CleanUpSystem : public System
	{
		public:
			CleanUpSystem();
			CleanUpSystem(std::vector<char> data);
			~CleanUpSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template <>
	const int SystemType<CleanUpSystem>::type = 2;

	template< typename T>
	struct IsCleanUpSystem { static bool value; };

	template<typename T>
	bool IsCleanUpSystem<T>::value = false;

	template<>
	bool IsCleanUpSystem<CleanUpSystem>::value = true;
	template<>
	bool IsSystem<CleanUpSystem>::value = true;
}

#endif