#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		private:
			//LineBuffer mBuffer;

			Material* mColorMat;
			Shader* mColorShader;

		public:
			DebugSystem();
			DebugSystem(std::vector<char> data);
			~DebugSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input, Time time);
	};

	template <>
	const int SystemType<DebugSystem>::type = 3;

	template< typename T>
	struct IsDebugSystem { static const bool value; };

	template<typename T>
	const bool IsDebugSystem<T>::value = false;

	template<>
	const bool IsDebugSystem<DebugSystem>::value = true;
	template<>
	const bool IsSystem<DebugSystem>::value = true;
	template<>
	const bool IsSystemInternal<DebugSystem>::value = true;
}

#endif