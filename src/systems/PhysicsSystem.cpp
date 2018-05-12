#include "PhysicsSystem.h"

#include "../core/Log.h"
#include "../core/Input.h"
#include "../core/Bounds.h"
#include "../core/Physics.h"

#include "../components/Transform.h"
// #include "../components/Fluid.h"
// #include "../components/Cloth.h"
// #include "../components/Particles.h"
// #include "../components/ParticleMesh.h"

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

	timestep = Physics::timestep;
	gravity = Physics::gravity;



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
	rigidbodies = manager->getRigidbodies();

	Physics::update(colliders);


	if(Input::getKeyDown(KeyCode::Tab)){
		start = true;
	}

	if(!start){
		return;
	}

	// gravity
	// for(unsigned int i = 0; i < rigidbodies.size(); i++){
	// 	if(rigidbodies[i]->useGravity){
	// 		Transform* transform = rigidbodies[i]->entity->getComponent<Transform>();

	// 		rigidbodies[i]->halfVelocity = rigidbodies[i]->velocity + 0.5f * timestep * glm::vec3(0.0f, -gravity, 0.0f);
	// 		transform->position += timestep * rigidbodies[i]->halfVelocity;
	// 		rigidbodies[i]->velocity = rigidbodies[i]->halfVelocity + 0.5f * timestep * glm::vec3(0.0f, -gravity, 0.0f);
	// 	}
	// }

	// spring joints
	std::vector<SpringJoint*> springJoints = manager->getSpringJoints();
	for(unsigned int i = 0; i < springJoints.size(); i++){
		float stiffness = springJoints[i]->stiffness;
		float damping = springJoints[i]->damping;
		glm::vec3 targetPosition = springJoints[i]->getTargetPosition();

		float fac1 = 1.0f - 0.5f * damping * timestep;
		float fac2 = 1.0f / (1.0f + 0.5f * damping * timestep);

		Transform* transform = springJoints[i]->entity->getComponent<Transform>();
		Rigidbody* rigidbody = springJoints[i]->entity->getComponent<Rigidbody>();

		Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);

		glm::vec3 position = transform->position;
		glm::vec3 halfVelocity = rigidbody->halfVelocity;
		glm::vec3 velocity = rigidbody->velocity;

		halfVelocity = fac1 * fac2 * halfVelocity - stiffness * timestep * fac2 * (position - targetPosition) + timestep * fac2 * (glm::vec3(0.0f, -gravity, 0.0f));
		position += position * timestep * halfVelocity;

		transform->position = position;
		rigidbody->halfVelocity = halfVelocity;

		Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);


		//Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);

		// rigidbody->halfVelocity = rigidbody->velocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
		// transform->position += timestep * rigidbody->halfVelocity;
		// rigidbody->velocity = rigidbody->halfVelocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
	
		//Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);
	}

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