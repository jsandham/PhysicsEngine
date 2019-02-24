#include "../../include/systems/ClothSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Cloth.h"

// #include <cuda.h>
// #include <cudagl.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <cuda_gl_interop.h>

using namespace PhysicsEngine;

ClothSystem::ClothSystem()
{
	type = 5;
}

ClothSystem::ClothSystem(std::vector<char> data)
{
	type = 5;
}

ClothSystem::~ClothSystem()
{
	
}

void* ClothSystem::operator new(size_t size)
{
	return getAllocator<ClothSystem>().allocate();
}

void ClothSystem::operator delete(void*)
{

}

void ClothSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfComponents<Cloth>(); i++){
		Cloth* cloth = world->getComponentByIndex<Cloth>(i);

		ClothDeviceData clothDeviceData;
		//fluidDeviceData.numBoids = fluid->numBoids;

		deviceData.push_back(ClothDeviceData());
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		allocateClothDeviceData(&deviceData[i]);
		initializeClothDeviceData(&deviceData[i]);
	}
}

void ClothSystem::update(Input input)
{
	for(size_t i = 0; i < deviceData.size(); i++){
		updateClothDeviceData(&deviceData[i]);
	}
}