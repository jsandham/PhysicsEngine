#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "Component.h"

namespace PhysicsEngine
{
	enum class LightType
	{
		Directional,
		Spot,
		Point,
		None
	};

	enum class ShadowType
	{
		Hard,
		Soft,
		None
	};

	class Light : public Component
	{
		public:
			LightType lightType;
			ShadowType shadowType;

		public:
			Light();
			virtual ~Light() = 0;
	};
}

#endif