#include "../../include/systems/System.h"

using namespace PhysicsEngine;

System::System()
{

}

System::~System()
{

}

void System::setSceneContext(SceneContext* context)
{
	this->context = context;
}