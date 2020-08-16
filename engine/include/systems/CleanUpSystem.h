#ifndef __CLEANUPSYSTEM_H__
#define __CLEANUPSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct CleanUpSystemHeader
	{
		Guid mSystemId;
		int32_t mUpdateOrder;
	};
#pragma pack(pop)

	class CleanUpSystem : public System
	{
		public:
			CleanUpSystem();
			CleanUpSystem(const std::vector<char>& data);
			~CleanUpSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(const std::vector<char>& data);

			void init(World* world);
			void update(Input input, Time time);
	};

	template <typename T>
	struct IsCleanUpSystem { static constexpr bool value = false; };

	template <>
	struct SystemType<CleanUpSystem> { static constexpr int type = PhysicsEngine::CLEANUPSYSTEM_TYPE;};
	template <>
	struct IsCleanUpSystem<CleanUpSystem> { static constexpr bool value = true; };
	template <>
	struct IsSystem<CleanUpSystem> { static constexpr bool value = true; };
	template <>
	struct IsSystemInternal<CleanUpSystem> { static constexpr bool value = true; };
}

#endif