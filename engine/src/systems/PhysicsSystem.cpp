#include <algorithm>

#include "../../include/systems/PhysicsSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Sphere.h"
#include "../../include/core/Triangle.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"
#include "../../include/core/UniformGrid.h"

#include "../../include/components/Transform.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/BoxCollider.h"
#include "../../include/components/MeshCollider.h"

using namespace PhysicsEngine;

PhysicsSystem::PhysicsSystem()
{

}

PhysicsSystem::PhysicsSystem(std::vector<char> data)
{
	deserialize(data);
}

PhysicsSystem::~PhysicsSystem()
{
	
}

std::vector<char> PhysicsSystem::serialize() const
{
	return serialize(systemId);
}

std::vector<char> PhysicsSystem::serialize(Guid systemId) const
{
	std::vector<char> data(sizeof(int));

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void PhysicsSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
}

void PhysicsSystem::init(World* world)
{
	this->world = world;

	UniformGrid* grid = world->getStaticPhysicsGrid();

	std::vector<BoundingSphere> boundingSpheres;
	std::vector<Sphere> spheres;
	std::vector<Bounds> bounds;
	std::vector<Triangle> triangles;

	for(int i = 0; i < world->getNumberOfComponents<SphereCollider>(); i++){
		SphereCollider* collider = world->getComponentByIndex<SphereCollider>(i);

		std::cout << "component id: " << collider->getId().toString() << std::endl;

		spheres.push_back(collider->sphere);

		BoundingSphere boundingSphere;
		boundingSphere.sphere = collider->sphere;
		boundingSphere.id = collider->getId();
		boundingSphere.primitiveType = 0;
		boundingSphere.index = i;

		boundingSpheres.push_back(boundingSphere);
	}

	for(int i = 0; i < world->getNumberOfComponents<BoxCollider>(); i++){
		BoxCollider* collider = world->getComponentByIndex<BoxCollider>(i);

		std::cout << "component id: " << collider->getId().toString() << std::endl;

		glm::vec3 min = collider->bounds.getMin();
		glm::vec3 max = collider->bounds.getMax();

		bounds.push_back(collider->bounds);

		BoundingSphere boundingSphere;
		boundingSphere.id = collider->getId();
		boundingSphere.sphere.centre = 0.5f * (min + max);
		boundingSphere.sphere.radius = glm::length(0.5f*(min - max));
		boundingSphere.primitiveType = 1;
		boundingSphere.index = i;

		boundingSpheres.push_back(boundingSphere);
	}

	// int index = 0;
	// for(int i = 0; i < world->getNumberOfComponents<MeshCollider>(); i++){
	// 	MeshCollider* collider = world->getComponentByIndex<MeshCollider>(i);
	// 	Transform* transform = world->getComponent<Transform>(collider->entityId);

	// 	glm::mat4 model = transform->getModelMatrix();

	// 	Mesh* mesh = world->getAsset<Mesh>(collider->meshId);

	// 	std::vector<float> vertices = mesh->vertices;
	// 	for(size_t j = 0; j < vertices.size() / 9; j++){
	// 		glm::vec4 v1 = model * glm::vec4(mesh->vertices[9*i], mesh->vertices[9*i + 1], mesh->vertices[9*i + 2], 1.0f);
	// 		glm::vec4 v2 = model * glm::vec4(mesh->vertices[9*i + 3], mesh->vertices[9*i + 4], mesh->vertices[9*i + 5], 1.0f);
	// 		glm::vec4 v3 = model * glm::vec4(mesh->vertices[9*i + 6], mesh->vertices[9*i + 7], mesh->vertices[9*i + 8], 1.0f);

	// 		Triangle triangle;
	// 		triangle.v1 = glm::vec3(v1.x, v1.y, v1.z);
	// 		triangle.v2 = glm::vec3(v2.x, v2.y, v2.z);
	// 		triangle.v3 = glm::vec3(v3.x, v3.y, v3.z);
			
	// 		triangles.push_back(triangle);

	// 		// compute bounding sphere
	// 		BoundingSphere boundingSphere;
	// 		boundingSphere.id = collider->componentId;
	// 		boundingSphere.sphere.centre = glm::vec3(0.0f, 0.0f, 0.0f);
	// 		boundingSphere.sphere.radius = 1.0f;
	// 		boundingSphere.primitiveType = 2;
	// 		boundingSphere.index = index;

	// 		index++;
	// 	}
	// }

	float minDiameter = 0.1f;

	for(size_t i = 0; i < boundingSpheres.size(); i++){
		if(2.0f * boundingSpheres[i].sphere.radius > minDiameter){
			minDiameter = 2.0f * boundingSpheres[i].sphere.radius;
		}
	}

	std::cout << "number of bounding spheres: " << boundingSpheres.size() << " number of spheres: " << spheres.size() << " number of bounds: " << bounds.size() << " number of triangles: " << triangles.size() << std::endl;

	Bounds worldBounds = *world->getWorldBounds();

	glm::ivec3 gridDim = glm::ivec3(worldBounds.size.x / minDiameter, worldBounds.size.y / minDiameter, worldBounds.size.z / minDiameter);

	gridDim.x = std::min(50, gridDim.x);
	gridDim.y = std::min(50, gridDim.y);
	gridDim.z = std::min(50, gridDim.z);

	std::cout << " min diameter: " << minDiameter << " gridDim.x: " << gridDim.x << " gridDim.y: " << gridDim.y << " gridDim.z: " << gridDim.z << std::endl;

	for(size_t i = 0; i < boundingSpheres.size(); i++){
		Sphere sphere = boundingSpheres[i].sphere;

		std::cout << "centre: " << sphere.centre.x << " " << sphere.centre.y << " " << sphere.centre.z << " " << sphere.radius << " id: " << boundingSpheres[i].id.toString() << std::endl;
	}

	//grid->create(worldBounds, gridDim, boundingSpheres, spheres, bounds, triangles);
}

void PhysicsSystem::update(Input input)
{
	// Octtree* stree = world->getStaticPhysicsTree();

	// stree->clear();

	// // rebuild dynamic octtree for physics raycasts
	// for(int i = 0; i < world->getNumberOfComponents<SphereCollider>(); i++){
	// 	SphereCollider* collider = world->getComponentByIndex<SphereCollider>(i);

	// 	stree->insert(collider->sphere, collider->componentId);
	// }






	// physics->tempClear();
	
	// // rebuild dynamic octtree for physics
	// for(int i = 0; i < world->getNumberOfComponents<SphereCollider>(); i++){
	// 	SphereCollider* collider = world->getComponentByIndex<SphereCollider>(i);
	// 	//std::cout << "collider: " << i << " centre: " << collider->sphere.centre.x << " " << collider->sphere.centre.y << " " << collider->sphere.centre.z << " radius: " << collider->sphere.radius << std::endl; 

	// 	physics->tempInsert(collider->sphere, collider->componentId);
	// }
}


















































// init
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





// update
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