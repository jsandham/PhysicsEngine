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
			Guid mSystemId;
			int mOrder;

			World* mWorld;

		public:
			System();
			virtual ~System() = 0;

			virtual std::vector<char> serialize() const = 0;
			virtual std::vector<char> serialize(Guid systemId) const = 0;
			virtual void deserialize(std::vector<char> data) = 0;

			virtual void init(World* world) = 0;
			virtual void update(Input input) = 0;

			Guid getId() const;
			int getOrder() const;

		private:
			friend class World;
	};

	template <typename T>
	struct SystemType { static const int type; };

	template <typename T>
	const int SystemType<T>::type = -1;

	template< typename T>
	struct IsSystem { static const bool value; };

	template<typename T>
	const bool IsSystem<T>::value = false;

	template<>
	const bool IsSystem<System>::value = true;
}

#endif