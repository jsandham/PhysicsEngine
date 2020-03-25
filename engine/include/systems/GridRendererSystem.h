#ifndef __GRID_RENDERER_SYSTEM_H__
#define __GRID_RENDERER_SYSTEM_H__

#include "System.h"

#include <vector>

namespace PhysicsEngine
{
	class GridRendererSystem : public System
	{
		public:
			GridRendererSystem();
			~GridRendererSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template <>
	const int SystemType<GridRendererSystem>::type = 4;

	template< typename T>
	struct IsGridRendererSystem { static const bool value; };

	template<typename T>
	const bool IsGridRendererSystem<T>::value = false;

	template<>
	const bool IsGridRendererSystem<GridRendererSystem>::value = true;
	template<>
	const bool IsSystem<GridRendererSystem>::value = true;
}

#endif