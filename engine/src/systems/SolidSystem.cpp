#include "../../include/systems/SolidSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Solid.h"

using namespace PhysicsEngine;

SolidSystem::SolidSystem()
{
	type = 7;
}

SolidSystem::SolidSystem(std::vector<char> data)
{
	type = 7;
}

SolidSystem::~SolidSystem()
{
	
}

void SolidSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfComponents<Solid>(); i++){
		Solid* solid = world->getComponentByIndex<Solid>(i);

		SolidDeviceData solidDeviceData;
		//fluidDeviceData.numBoids = fluid->numBoids;

		deviceData.push_back(SolidDeviceData());
	}

	for(size_t i = 0; i < deviceData.size(); i++){
		allocateSolidDeviceData(&deviceData[i]);
		initializeSolidDeviceData(&deviceData[i]);
	}
}

void SolidSystem::update(Input input)
{
	for(size_t i = 0; i < deviceData.size(); i++){
		updateSolidDeviceData(&deviceData[i]);
	}
}