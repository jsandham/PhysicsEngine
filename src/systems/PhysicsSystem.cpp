#include "PhysicsSystem.h"

#include "../core/Bounds.h"
#include "../core/Physics.h"

#include "../components/Fluid.h"
#include "../components/Cloth.h"
#include "../components/Particles.h"
#include "../components/ParticleMesh.h"

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem(Manager *manager)
{
	this->manager = manager;
}

PhysicsSystem::~PhysicsSystem()
{

}

void PhysicsSystem::init()
{
	Bounds bounds(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(20.0f, 20.0f, 20.0f));

	Physics::init(bounds, 2);
	/*std::vector<Fluid*> fluids = manager->getFluids();
	std::vector<Cloth*> cloths = manager->getCloths();

	for (unsigned int i = 0; i < fluids.size(); i++){
		fluids[i]->init();
	}

	for (unsigned int i = 0; i < cloths.size(); i++){
		cloths[i]->init();
	}*/

	/*std::map<int, Entity*>::iterator it;
	for (it = Entity::entities.begin(); it != Entity::entities.end(); it++){
		Entity *entity = it->second;

		Collider *collider = entity->GetComponent<Collider>();
		ParticleMesh *mesh = entity->GetComponent<ParticleMesh>();
		FluidParticles *fp = entity->GetComponent<FluidParticles>();
		Particles *p = entity->GetComponent<Particles>();
		ClothParticles *cp = entity->GetComponent<ClothParticles>();

		if (collider != NULL){
			colliders.push_back(collider);
		}

		if (mesh == NULL){
			std::cout << "PhysicsSystem: Entity must contain a ParticleMesh component. Ignoring this entity" << std::endl;
			continue;
		}

		if (fp != NULL){
			particles.push_back(fp);
			particleMeshes.push_back(mesh);
		}

		if (p != NULL){
			particles.push_back(p);
			particleMeshes.push_back(mesh);
		}

		if (cp != NULL){
			particles.push_back(cp);
			particleMeshes.push_back(mesh);
		}
	}

	for (unsigned int i = 0; i < particles.size(); i++){
		particles[i]->init();
	}*/
}

void PhysicsSystem::update()
{
	colliders = manager->getColliders();

	Physics::update(colliders);

	//std::vector<Fluid*> fluids = manager->getFluids();
	//std::vector<Cloth*> cloths = manager->getCloths();

	//for (unsigned int i = 0; i < fluids.size(); i++){
	//	fluids[i]->update();

	//	//particleMeshes[i]->setPoints(particles[i]->getParticles());
	//}

	//for (unsigned int i = 0; i < cloths.size(); i++){
	//	cloths[i]->update();

	//	//particleMeshes[i]->setPoints(particles[i]->getParticles());
	//}
}