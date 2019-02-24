#include "../../include/systems/BoidsSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Boids.h"

// #include <cuda.h>
// #include <cudagl.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <cuda_gl_interop.h>

using namespace PhysicsEngine;

BoidsSystem::BoidsSystem()
{
	type = 4;
}

BoidsSystem::BoidsSystem(std::vector<char> data)
{
	type = 4;
}

BoidsSystem::~BoidsSystem()
{
	
}

void* BoidsSystem::operator new(size_t size)
{
	return getAllocator<BoidsSystem>().allocate();
}

void BoidsSystem::operator delete(void*)
{

}

void BoidsSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfComponents<Boids>(); i++){
		Boids* boids = world->getComponentByIndex<Boids>(i);

		BoidsDeviceData boidsDeviceData;
		boidsDeviceData.numBoids = boids->numBoids;

		deviceData.push_back(BoidsDeviceData());
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		allocateBoidsDeviceData(&deviceData[i]);
		initializeBoidsDeviceData(&deviceData[i]);
	}
}

void BoidsSystem::update(Input input)
{
	for(size_t i = 0; i < deviceData.size(); i++){
		updateBoidsDeviceData(&deviceData[i]);
	}
}


// init
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

//for (unsigned int i = 0; i < cloths.size(); i++){
//	cloths[i]->init();
//}





// update
// for(unsigned int i = 0; i < cudaCloths.size(); i++){
// 	CudaPhysics::update(&cudaCloths[i]);
// }


//std::vector<Cloth*> cloths = manager->getCloths();

//for (unsigned int i = 0; i < cloths.size(); i++){
//	cloths[i]->update();

//	//particleMeshes[i]->setPoints(particles[i]->getParticles());
//}