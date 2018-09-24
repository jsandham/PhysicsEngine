#include "../../include/systems/System.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

System::System()
{

}

System::~System()
{

}

void System::setManager(Manager* manager)
{
	this->manager = manager;
}

void System::setSceneContext(SceneContext* context)
{
	this->context = context;
}