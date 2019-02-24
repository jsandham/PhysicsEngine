#include "../../include/systems/FluidSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Fluid.h"

// #include <cuda.h>
// #include <cudagl.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <cuda_gl_interop.h>

using namespace PhysicsEngine;

FluidSystem::FluidSystem()
{
	type = 6;
}

FluidSystem::FluidSystem(std::vector<char> data)
{
	type = 6;
}

FluidSystem::~FluidSystem()
{
	
}

void* FluidSystem::operator new(size_t size)
{
	return getAllocator<FluidSystem>().allocate();
}

void FluidSystem::operator delete(void*)
{

}

void FluidSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfComponents<Fluid>(); i++){
		Fluid* fluid = world->getComponentByIndex<Fluid>(i);

		FluidDeviceData fluidDeviceData;
		//fluidDeviceData.numBoids = fluid->numBoids;

		deviceData.push_back(FluidDeviceData());
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		allocateFluidDeviceData(&deviceData[i]);
		initializeFluidDeviceData(&deviceData[i]);
	}
}

void FluidSystem::update(Input input)
{
	for(size_t i = 0; i < deviceData.size(); i++){
		updateFluidDeviceData(&deviceData[i]);
	}
}