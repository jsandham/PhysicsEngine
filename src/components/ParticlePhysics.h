#ifndef __PARTICLEPHYSICS_H__
#define __PARTICLEPHYSICS_H__

#include <vector>

#include "../glm/glm.hpp"

#include "Component.h"

namespace PhysicsEngine
{
	class ParticlePhysics : public Component
	{
		public:
			ParticlePhysics();
			virtual ~ParticlePhysics() = 0;

			virtual void init() = 0;
			virtual void update() = 0;

			virtual void setParticles(std::vector<float> &particles) = 0;
			virtual void setParticleTypes(std::vector<int> &particleTypes) = 0;
			virtual std::vector<float>& getParticles() = 0;
			virtual std::vector<int>& getParticleTypes() = 0;
	};
}

#endif