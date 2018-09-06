#include "../../include/systems/PhysicsSystem.h"

#include "../../include/core/Log.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Fluid.h"
#include "../../include/components/Cloth.h"
#include "../../include/components/Solid.h"

#include "../../include/cuda/CudaPhysics.cuh"

#include <cuda.h>
#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem(Manager *manager, SceneContext* context)
{
	this->manager = manager;
}

PhysicsSystem::~PhysicsSystem()
{
	// for(unsigned int i = 0; i < cudaCloths.size(); i++){
	// 	CudaPhysics::deallocate(&cudaCloths[i]);
	// }

	// for(unsigned int i = 0; i < cudaSolids.size(); i++){
	// 	CudaPhysics::deallocate(&cudaSolids[i]);
	// }

	// for(unsigned int i = 0; i < cudaFluids.size(); i++){
	// 	CudaPhysics::deallocate(&cudaFluids[i]);
	// }
}

void PhysicsSystem::init()
{
	// Bounds bounds(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(20.0f, 20.0f, 20.0f));

	// Physics::init(bounds, 2);

	// timestep = Physics::timestep;
	// gravity = Physics::gravity;

	// std::vector<Cloth*> cloths = manager->getCloths();
	// for(unsigned int i = 0; i < cloths.size(); i++){

	// 	cudaCloths.push_back(CudaCloth());

	// 	cudaCloths[i].nx = cloths[i]->nx;
	// 	cudaCloths[i].ny = cloths[i]->ny;
	// 	cudaCloths[i].particles = cloths[i]->particles;
	// 	cudaCloths[i].particleTypes = cloths[i]->particleTypes;

	// 	cudaCloths[i].dt = timestep;
	// 	cudaCloths[i].kappa = cloths[i]->kappa;
	// 	cudaCloths[i].c = cloths[i]->c;
	// 	cudaCloths[i].mass = cloths[i]->mass;

	// 	CudaPhysics::allocate(&cudaCloths[i]);

	// 	cloths[i]->clothVAO.generate();
	// 	cloths[i]->clothVAO.bind();
	// 	cloths[i]->vertexVBO.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	// 	cloths[i]->vertexVBO.bind();
	// 	cloths[i]->vertexVBO.setData(NULL, 9*2*(cloths[i]->nx-1)*(cloths[i]->ny-1)*sizeof(float)); 
	// 	cloths[i]->clothVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	// 	cloths[i]->normalVBO.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	// 	cloths[i]->normalVBO.bind();
	// 	cloths[i]->normalVBO.setData(NULL, 9*2*(cloths[i]->nx-1)*(cloths[i]->ny-1)*sizeof(float)); 
	// 	cloths[i]->clothVAO.setLayout(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
	// 	cloths[i]->clothVAO.unbind();

	// 	cudaGraphicsGLRegisterBuffer(&(cudaCloths[i].cudaVertexVBO), cloths[i]->vertexVBO.handle, cudaGraphicsMapFlagsWriteDiscard);
	// 	cudaGraphicsGLRegisterBuffer(&(cudaCloths[i].cudaNormalVBO), cloths[i]->normalVBO.handle, cudaGraphicsMapFlagsWriteDiscard);

	// 	CudaPhysics::initialize(&cudaCloths[i]);
	// }

	// std::vector<Solid*> solids = manager->getSolids();
	// for(unsigned int i = 0; i < solids.size(); i++){
	// 	cudaSolids.push_back(CudaSolid());

	// 	cudaSolids[i].c = solids[i]->c;                   
	//     cudaSolids[i].rho = solids[i]->rho;                           
	//     cudaSolids[i].Q = solids[i]->Q;    
	//     cudaSolids[i].k = solids[i]->k;   

	// 	cudaSolids[i].dim = solids[i]->dim; 
	//     cudaSolids[i].ng = solids[i]->ng;  
	//     cudaSolids[i].n = solids[i]->n;                     
	//     cudaSolids[i].nte = solids[i]->nte;        
	//     cudaSolids[i].ne = solids[i]->ne;                   
	//     cudaSolids[i].ne_b = solids[i]->ne_b;                                
	//     cudaSolids[i].npe = solids[i]->npe;      
	//     cudaSolids[i].npe_b = solids[i]->npe_b;      
	//     cudaSolids[i].type = solids[i]->type;                        
	//     cudaSolids[i].type_b = solids[i]->type_b; 

	// 	cudaSolids[i].vertices = solids[i]->vertices;
	// 	cudaSolids[i].connect = solids[i]->connect;
	// 	cudaSolids[i].bconnect = solids[i]->bconnect;
	// 	cudaSolids[i].groups = solids[i]->groups;

	// 	CudaPhysics::allocate(&cudaSolids[i]);

	// 	solids[i]->solidVAO.generate();
	// 	solids[i]->solidVAO.bind();
	// 	solids[i]->vertexVBO.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	// 	solids[i]->vertexVBO.bind();
	// 	solids[i]->vertexVBO.setData(NULL, 3*(solids[i]->ne_b)*(solids[i]->npe_b)*sizeof(float)); 
	// 	solids[i]->solidVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	// 	solids[i]->normalVBO.generate(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	// 	solids[i]->normalVBO.bind();
	// 	solids[i]->normalVBO.setData(NULL, 3*(solids[i]->ne_b)*(solids[i]->npe_b)*sizeof(float)); 
	// 	solids[i]->solidVAO.setLayout(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
	// 	solids[i]->solidVAO.unbind();

	// 	cudaGraphicsGLRegisterBuffer(&(cudaSolids[i].cudaVertexVBO), solids[i]->vertexVBO.handle, cudaGraphicsMapFlagsWriteDiscard);
	// 	cudaGraphicsGLRegisterBuffer(&(cudaSolids[i].cudaNormalVBO), solids[i]->normalVBO.handle, cudaGraphicsMapFlagsWriteDiscard);

	// 	CudaPhysics::initialize(&cudaSolids[i]);
	// }

	// std::vector<Fluid*> fluids = manager->getFluids();
	// for(unsigned int i = 0; i < fluids.size(); i++){
	// 	cudaFluids.push_back(CudaFluid());

	// 	CudaPhysics::allocate(&cudaFluids[i]);
	// 	CudaPhysics::initialize(&cudaFluids[i]);
	// }

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

		Collider *collider = entity->GetComponent<Collider>(manager);
		ParticleMesh *mesh = entity->GetComponent<ParticleMesh>(manager);
		FluidParticles *fp = entity->GetComponent<FluidParticles>(manager);
		Particles *p = entity->GetComponent<Particles>(manager);
		ClothParticles *cp = entity->GetComponent<ClothParticles>(manager);

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
	// colliders = manager->getColliders();
	// rigidbodies = manager->getRigidbodies();

	// Physics::update(colliders);


	// if(Input::getKeyDown(KeyCode::Tab)){
	// 	start = true;
	// }

	// if(!start){
	// 	return;
	// }

	// gravity
	// for(unsigned int i = 0; i < rigidbodies.size(); i++){
	// 	if(rigidbodies[i]->useGravity){
	// 		Transform* transform = rigidbodies[i]->entity->getComponent<Transform>(manager);

	// 		rigidbodies[i]->halfVelocity = rigidbodies[i]->velocity + 0.5f * timestep * glm::vec3(0.0f, -gravity, 0.0f);
	// 		transform->position += timestep * rigidbodies[i]->halfVelocity;
	// 		rigidbodies[i]->velocity = rigidbodies[i]->halfVelocity + 0.5f * timestep * glm::vec3(0.0f, -gravity, 0.0f);
	// 	}
	// }

	// spring joints
	// for(int t = 0; t < 10; t++){
	// 	std::vector<SpringJoint*> springJoints = manager->getSpringJoints();
	// 	for(unsigned int i = 0; i < springJoints.size(); i++){
	// 		float stiffness = springJoints[i]->stiffness;
	// 		float damping = springJoints[i]->damping;
	// 		float fac1 = 1.0f - 0.5f * damping * timestep;
	// 		float fac2 = 1.0f / (1.0f + 0.5f * damping * timestep);

	// 		Transform* transform = springJoints[i]->getEntity(manager->getEntities())->getComponent<Transform>(manager->getTransforms());
	// 		Rigidbody* rigidbody = springJoints[i]->getEntity(manager->getEntities())->getComponent<Rigidbody>(manager->getRigidbodies());

	// 		glm::vec3 targetPosition = springJoints[i]->getTargetPosition();

	// 		glm::vec3 position = transform->position;
	// 		glm::vec3 halfVelocity = rigidbody->halfVelocity;
	// 		glm::vec3 velocity = rigidbody->velocity;

	// 		halfVelocity = fac1 * fac2 * halfVelocity - stiffness * timestep * fac2 * (position - targetPosition) + timestep * fac2 * (glm::vec3(0.0f, -gravity, 0.0f));
	// 		position += timestep * halfVelocity;

	// 		transform->position = position;
	// 		rigidbody->halfVelocity = halfVelocity;

	// 		// rigidbody->halfVelocity = rigidbody->velocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
	// 		// transform->position += timestep * rigidbody->halfVelocity;
	// 		// rigidbody->velocity = rigidbody->halfVelocity + 0.5f * timestep * 100 * (glm::vec3(4.0f, 4.0f, 4.0f) - transform->position + glm::vec3(0.0f, -gravity, 0.0f));
		
	// 		//Log::Info("iteration: %d velocity %f %f %f", i, rigidbody->velocity.x, rigidbody->velocity.y, rigidbody->velocity.z);
	// 	}
	// }

	// hinge joints


	// cloth
	// for(unsigned int i = 0; i < cudaCloths.size(); i++){
	// 	CudaPhysics::update(&cudaCloths[i]);
	// }

	// // fluid
	// for(unsigned int i = 0; i < cudaFluids.size(); i++){
	// 	CudaPhysics::update(&cudaFluids[i]);
	// }

	// // fem
	// for(unsigned int i = 0; i < cudaSolids.size(); i++){
	// 	CudaPhysics::update(&cudaSolids[i]);
	// }


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