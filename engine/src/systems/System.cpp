#include "../../include/systems/System.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

System::System()
{

}

System::~System()
{

}

void System::setWorld(World* world)
{
	this->world = world;
}

void System::setSceneContext(SceneContext* context)
{
	this->context = context;
}