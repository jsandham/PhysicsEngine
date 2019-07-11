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
	
}

ClothSystem::ClothSystem(std::vector<char> data)
{
	deserialize(data);
}

ClothSystem::~ClothSystem()
{
	
}

std::vector<char> ClothSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void ClothSystem::deserialize(std::vector<char> data)
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