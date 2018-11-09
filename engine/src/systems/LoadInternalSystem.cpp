#include <iostream>

#include "../../include/systems/LoadInternalSystem.h"
#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"

using namespace	PhysicsEngine;

System* PhysicsEngine::loadInternalSystem(unsigned char* data)
{
	int type = *reinterpret_cast<int*>(data);

	if(type == 0){
		return new RenderSystem(data);
	}
	else if(type == 1){
		return new PhysicsSystem(data);
	}
	else if(type == 2){
		return new CleanUpSystem(data);
	}
	else if(type == 3){
		return new DebugSystem(data);
	}
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load internal system" << std::endl;
		return NULL;
	}
}