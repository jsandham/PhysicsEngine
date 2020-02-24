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

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template <>
	const int SystemType<GridRendererSystem>::type = 4;

	template< typename T>
	struct IsGridRendererSystem { static bool value; };

	template<typename T>
	bool IsGridRendererSystem<T>::value = false;

	template<>
	bool IsGridRendererSystem<GridRendererSystem>::value = true;
	template<>
	bool IsSystem<GridRendererSystem>::value = true;
}

#endif