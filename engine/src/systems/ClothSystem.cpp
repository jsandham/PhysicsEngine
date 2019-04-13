#include "../../include/systems/ClothSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Input.h"
#include "../../include/core/Bounds.h"
#include "../../include/core/Physics.h"
#include "../../include/core/World.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Cloth.h"

using namespace PhysicsEngine;

ClothSystem::ClothSystem()
{
	type = 5;
}

ClothSystem::ClothSystem(std::vector<char> data)
{
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if(type != 5){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
}

ClothSystem::~ClothSystem()
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