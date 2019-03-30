#include <math.h>

#include "../../include/systems/BoidsSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Camera.h"
#include "../../include/components/Boids.h"

#include "../../include/graphics/Graphics.h"

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

		Mesh* mesh = world->getAsset<Mesh>(boids->meshId);
		// Material* material = world->getAsset<Material>(boids->materialId);
		Shader* shader = world->getAsset<Shader>(boids->shaderId);

		if(mesh != NULL && shader != NULL){
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

			shader->compile();

			if(shader->isCompiled()){
				std::cout << "AAAAAAAAAAA shader compiled successfully" << std::endl;
			}

			boidsDeviceData.mesh = mesh;
			// boidsDeviceData.material = material;
			boidsDeviceData.shader = shader;

			glGenVertexArrays(1, &boidsDeviceData.VAO);
			glBindVertexArray(boidsDeviceData.VAO);

			glGenBuffers(1, &boidsDeviceData.vertexVBO);
			glBindBuffer(GL_ARRAY_BUFFER, boidsDeviceData.vertexVBO);
			glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size()*sizeof(float), &(mesh->vertices[0]), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glGenBuffers(1, &boidsDeviceData.normalVBO);
			glBindBuffer(GL_ARRAY_BUFFER, boidsDeviceData.normalVBO);
			glBufferData(GL_ARRAY_BUFFER, mesh->normals.size()*sizeof(float), &(mesh->normals[0]), GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

			glGenBuffers(1, &boidsDeviceData.texCoordVBO);
			glBindBuffer(GL_ARRAY_BUFFER, boidsDeviceData.texCoordVBO);
			glBufferData(GL_ARRAY_BUFFER, mesh->texCoords.size()*sizeof(float), &(mesh->texCoords[0]), GL_STATIC_DRAW);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

			glGenBuffers(1, &boidsDeviceData.instanceModelVBO);
			glBindBuffer(GL_ARRAY_BUFFER, boidsDeviceData.instanceModelVBO);
			glBufferData(GL_ARRAY_BUFFER, boidsDeviceData.numBoids * sizeof(glm::mat4), NULL, GL_DYNAMIC_DRAW);
			glEnableVertexAttribArray(3);
	        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
	        glEnableVertexAttribArray(4);
	        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)(sizeof(glm::vec4)));
	        glEnableVertexAttribArray(5);
	        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)(2 * sizeof(glm::vec4)));
	        glEnableVertexAttribArray(6);
	        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)(3 * sizeof(glm::vec4)));

	        glVertexAttribDivisor(3, 1);
	        glVertexAttribDivisor(4, 1);
	        glVertexAttribDivisor(5, 1);
	        glVertexAttribDivisor(6, 1);

			glBindVertexArray(0);

			deviceData.push_back(boidsDeviceData);
		}
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		allocateBoidsDeviceData(&deviceData[i]);
		initializeBoidsDeviceData(&deviceData[i]);
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Boids System failed with error code: " << error << " during initialization" << std::endl;;
	}
}

void BoidsSystem::update(Input input)
{
	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		updateBoidsDeviceData(&deviceData[i]);

		//std::cout << "boids vao: " << deviceData[i].VAO << " number of boids: " << deviceData[i].numBoids << std::endl;

		glBindVertexArray(deviceData[i].VAO);
		glBindBuffer(GL_ARRAY_BUFFER, deviceData[i].instanceModelVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, deviceData[i].numBoids*sizeof(glm::mat4), deviceData[i].h_modelMatrices);

		Graphics::use(deviceData[i].shader);
		Graphics::setMat4(deviceData[i].shader, "projection", camera->getProjMatrix());
		Graphics::setMat4(deviceData[i].shader, "view", camera->getViewMatrix());

		glDrawArraysInstanced(GL_TRIANGLES, 0, (GLsizei)deviceData[i].mesh->vertices.size() / 3, deviceData[i].numBoids);
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Boids System failed with error code: " << error << " during update" << std::endl;;
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