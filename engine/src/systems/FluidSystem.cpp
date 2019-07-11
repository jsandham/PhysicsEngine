#include "../../include/systems/FluidSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Fluid.h"

using namespace PhysicsEngine;

FluidSystem::FluidSystem()
{
	
}

FluidSystem::FluidSystem(std::vector<char> data)
{
	deserialize(data);
}

FluidSystem::~FluidSystem()
{
	
}

std::vector<char> FluidSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void FluidSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
	// size_t index = sizeof(char);
	// type = *reinterpret_cast<int*>(&data[index]);
	// index += sizeof(int);
	// order = *reinterpret_cast<int*>(&data[index]);

	// if(type != 0){
	// 	std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	// }
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