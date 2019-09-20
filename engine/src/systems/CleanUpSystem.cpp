#include "../../include/systems/CleanUpSystem.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

CleanUpSystem::CleanUpSystem()
{
	
}

CleanUpSystem::CleanUpSystem(std::vector<char> data)
{
	deserialize(data);
}

CleanUpSystem::~CleanUpSystem()
{

}

std::vector<char> CleanUpSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void CleanUpSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
}

void CleanUpSystem::init(World* world)
{
	this->world = world;
}

void CleanUpSystem::update(Input input)
{
	world->clearIdsMarkedMoved();

	std::vector<triple<Guid, Guid, int>> componentIdsMarkedLatentDestroy = world->getComponentIdsMarkedLatentDestroy();
	for(size_t i = 0; i < componentIdsMarkedLatentDestroy.size(); i++){
		//std::cout << "clean up destroying component: " << componentIdsMarkedLatentDestroy[i].first.toString() << " " << componentIdsMarkedLatentDestroy[i].second.toString() << " " << componentIdsMarkedLatentDestroy[i].third << std::endl;
		
		Log::info("Number of camera before: \n");
		int cameraCount = world->getNumberOfComponents<Camera>();
		Log::info(&std::to_string(cameraCount)[0]);

		std::string message = "clean up destroying component: " + componentIdsMarkedLatentDestroy[i].first.toString() + " " + componentIdsMarkedLatentDestroy[i].second.toString() + " " + std::to_string(componentIdsMarkedLatentDestroy[i].third) + "\n";
		Log::info(&message[0]);

		world->immediateDestroyComponent(componentIdsMarkedLatentDestroy[i].first, componentIdsMarkedLatentDestroy[i].second, componentIdsMarkedLatentDestroy[i].third);

		Log::info("Number of camera after: \n");
		cameraCount = world->getNumberOfComponents<Camera>();
		Log::info(&std::to_string(cameraCount)[0]);
	}

	std::vector<Guid> entityIdsMarkedForLatentDestroy = world->getEntityIdsMarkedLatentDestroy();
	for(size_t i = 0; i < entityIdsMarkedForLatentDestroy.size(); i++){
		//std::cout << "clean up destroying entity: " << entityIdsMarkedForLatentDestroy[i].toString() << std::endl;
		std::string message = "clean up destroying entity: " + entityIdsMarkedForLatentDestroy[i].toString() + "\n";
		Log::info(&message[0]);
		world->immediateDestroyEntity(entityIdsMarkedForLatentDestroy[i]);
	}

	world->clearIdsMarkedCreatedOrDestroyed();
}