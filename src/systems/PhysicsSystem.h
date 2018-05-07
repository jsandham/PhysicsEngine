#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include "System.h"

#include "../memory/Manager.h"

#include "../components/ParticlePhysics.h"
#include "../components/ParticleMesh.h"
#include "../components/Collider.h"

namespace PhysicsEngine
{
	class PhysicsSystem : public System
	{
		private:
			std::vector<Collider*> colliders;
			//std::vector<ParticlePhysics*> particles;
			//std::vector<ParticleMesh*> particleMeshes;*/

		public:
			PhysicsSystem(Manager *manager);
			~PhysicsSystem();

			void init();
			void update();
	};
}

#endif