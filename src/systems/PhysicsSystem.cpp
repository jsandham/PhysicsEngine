#include "PhysicsSystem.h"

#include "../core/Log.h"
#include "../core/Input.h"
#include "../core/Bounds.h"
#include "../core/Physics.h"

#include "../components/Transform.h"
#include "../components/Fluid.h"
#include "../components/Cloth.h"

#include "../cuda/CudaPhysics.cuh"

#include <cuda.h>
#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem(Manager *manager)
{
	this->manager = manager;
}

PhysicsSystem::~PhysicsSystem()
{
	for(unsigned int i = 0; i < cudaCloths.size(); i++){
		CudaPhysics::deallocate(&cudaCloths[i]);
	}

	for(unsigned int i = 0; i < cudaFluids.size(); i++){
		CudaPhysics::deallocate(&cudaFluids[i]);
	}
}

void PhysicsSystem::init()
{
	Bounds bounds(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(20.0f, 20.0f, 20.0f));

	Physics::init(bounds, 2);

	timestep = Physics::timestep;
	gravity = Physics::gravity;

	std::vector<Cloth*> cloths = manager->getCloths();
	for(unsigned int i = 0; i < cloths.size(); i++){

		cudaCloths.push_back(CudaCloth());

		cudaCloths[i].nx = cloths[i]->nx;
		cudaCloths[i].ny = cloths[i]->ny;
		cudaCloths[i].particles = cloths[i]->particles;
		cudaCloths[i].particleTypes = cloths[i]->particleTypes;

		cudaCloths[i].dt = timestep;
		cudaCloths[i].kappa = cloths[i]->kappa;
		cudaCloths[i].c = cloths[i]->c;
		cudaCloths[i].mass = cloths[i]->mass;

		CudaPhysics::allocate(&cudaCloths[i]);

		cloths[i]->vao.generate();
		cloths[i]->vao.bind();
		cloths[i]->vbo.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		cloths[i]->vbo.bind();
		// cloths[i]->vbo.setData(&(cloths[i]->particles[0]), 3*cloths[i]->nx*cloths[i]->ny*sizeof(float)); 
		//cloths[i]->vbo.setData(NULL, 3*cloths[i]->nx*cloths[i]->ny*sizeof(float)); 
		cloths[i]->vbo.setData(NULL, 9*2*(cloths[i]->nx-1)*(cloths[i]->ny-1)*sizeof(float)); 
		cloths[i]->vao.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
		cloths[i]->vao.unbind();

		cudaGraphicsGLRegisterBuffer(&(cudaCloths[i].vbo_cuda), cloths[i]->vbo.handle, cudaGraphicsMapFlagsWriteDiscard);

		CudaPhysics::initialize(&cudaCloths[i]);
	}

	std::vector<FESolid*> fesolids = manager->getFESolids();
	for(unsigned int i = 0; i < fesolids.size(); i++){
		cudaFEMs.push_back(CudaFEM());

		cudaFEMs[i].vertices = fesolids[i]->vertices;
		cudaFEMs[i].connect = fesolids[i]->connect;
		cudaFEMs[i].bconnect = fesolids[i]->bconnect;
		cudaFEMs[i].groups = fesolids[i]->groups;

		CudaPhysics::allocate(&cudaFEMs[i]);

		fesolids[i]->vao.generate();
		fesolids[i]->vao.bind();
		fesolids[i]->vbo.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		fesolids[i]->vbo.bind();
		//fesolids[i]->vbo.setData(NULL, 9*2*(fesolids[i]->nx-1)*(fesolids[i]->ny-1)*sizeof(float)); 
		fesolids[i]->vao.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
		fesolids[i]->vao.unbind();

		cudaGraphicsGLRegisterBuffer(&(cudaFEMs[i].vbo_cuda), fesolids[i]->vbo.handle, cudaGraphicsMapFlagsWriteDiscard);

		CudaPhysics::initialize(&cudaFEMs[i]);
	}

	std::vector<Fluid*> fluids = manager->getFluids();
	for(unsigned int i = 0; i < fluids.size(); i++){
		cudaFluids.push_back(CudaFluid());

		CudaPhysics::allocate(&cudaFluids[i]);
		CudaPhysics::initialize(&cudaFluids[i]);
	}

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
	for(int t = 0; t < 10; t++){
		std::vector<SpringJoint*> springJoints = manager->getSpringJoints();
		for(unsigned int i = 0; i < springJoints.size(); i++){
			float stiffness = springJoints[i]->stiffness;
			float damping = springJoints[i]->damping;
			float fac1 = 1.0f - 0.5f * damping * timestep;
			float fac2 = 1.0f / (1.0f + 0.5f * damping * timestep);

			Transform* transform = springJoints[i]->entity->getComponent<Transform>();
			Rigidbody* rigidbody = springJoints[i]->entity->getComponent<Rigidbody>();

			glm::vec3 targetPosition = springJoints[i]->getTargetPosition();

			glm::vec3 position = transform->position;
			glm::vec3 halfVelocity = rigidbody->halfVelocity;
			glm::vec3 velocity = rigidbody->velocity;

			halfVelocity = fac1 * fac2 * halfVelocity - stiffness * timestep * fac2 * (position - targetPosition) + timestep * fac2 * (glm::vec3(0.0f, -gravity, 0.0f));
			position += timestep * halfVelocity;

			transform->position = position;
			rigidbody->halfVelocity = halfVelocity;

			// rigidbody->halfVelocity = rigidbody->velocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
			// transform->position += timestep * rigidbody->halfVelocity;
			// rigidbody->velocity = rigidbody->halfVelocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
		
			//Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);
		}
	}

	// hinge joints


	// cloth
	for(unsigned int i = 0; i < cudaCloths.size(); i++){
		CudaPhysics::update(&cudaCloths[i]);
	}

	// fluid
	for(unsigned int i = 0; i < cudaFluids.size(); i++){
		CudaPhysics::update(&cudaFluids[i]);
	}

	// fem
	for(unsigned int i = 0; i < cudaFEMs.size(); i++){
		CudaPhysics::update(&cudaFEMs[i]);
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