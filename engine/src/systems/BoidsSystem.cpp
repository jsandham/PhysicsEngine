#include <math.h>

#include "../../include/systems/BoidsSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Boids.h"

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

void BoidsSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfComponents<Boids>(); i++){
		Boids* boids = world->getComponentByIndex<Boids>(i);

		glm::vec3 boundsSize = boids->bounds.size;
		float h = boids->h;

		int voxelXDim = (int)ceil(boundsSize.x / h);
		int voxelYDim = (int)ceil(boundsSize.y / h);
		int voxelZDim = (int)ceil(boundsSize.z / h);

		glm::vec3 voxelGridSize = glm::vec3(h * voxelXDim, h * voxelYDim, h * voxelZDim);

		int numVoxels = voxelXDim * voxelYDim * voxelZDim;

		BoidsDeviceData boidsDeviceData;
		boidsDeviceData.numBoids = boids->numBoids;
		boidsDeviceData.numVoxels = numVoxels;
		boidsDeviceData.h = h;
		boidsDeviceData.voxelGridDim.x = voxelXDim;
		boidsDeviceData.voxelGridDim.y = voxelYDim;
		boidsDeviceData.voxelGridDim.z = voxelZDim;
		boidsDeviceData.voxelGridSize.x = voxelGridSize.x;
		boidsDeviceData.voxelGridSize.y = voxelGridSize.y;
		boidsDeviceData.voxelGridSize.z = voxelGridSize.z;

		std::cout << "numBoids: " << boidsDeviceData.numBoids << " numVoxels: " << numVoxels << " h: " << h << " voxel grid dim: " << voxelXDim << " " << voxelYDim << " " << voxelZDim << " voxel grid size: " << voxelGridSize.x << " " << voxelGridSize.y << " " << voxelGridSize.z << std::endl;


		deviceData.push_back(boidsDeviceData);
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