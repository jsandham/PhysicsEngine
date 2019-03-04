#ifndef __BOID_H__
#define __BOID_H__

#include <vector>

#include "Component.h"

#include "../core/Bounds.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct BoidsHeader
	{
		Guid componentId;
		Guid entityId;
		Guid meshId;
		Guid materialId;
		int numBoids;
		float h;
		Bounds bounds;
	};
#pragma pack(pop)

	class Boids : public Component
	{
		public:
			Guid meshId;
			Guid materialId;

			int numBoids;
			float h;
			Bounds bounds;

		public:
			Boids();
			Boids(std::vector<char> data);
			~Boids();
	};
}

#endif