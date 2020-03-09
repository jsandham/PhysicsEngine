#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"
#include "../graphics/Graphics.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		private:
			LineBuffer buffer;

			Material* colorMat;
			Shader* colorShader;

		public:
			DebugSystem();
			DebugSystem(std::vector<char> data);
			~DebugSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};

	template <>
	const int SystemType<DebugSystem>::type = 3;

	template< typename T>
	struct IsDebugSystem { static bool value; };

	template<typename T>
	bool IsDebugSystem<T>::value = false;

	template<>
	bool IsDebugSystem<DebugSystem>::value = true;
	template<>
	bool IsSystem<DebugSystem>::value = true;
}

#endif