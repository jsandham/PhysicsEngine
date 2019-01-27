#include "stdafx.h"
#include <iostream>

#include <systems/LoadSystem.h>

#include "../include/LogicSystem.h"
#include "../include/PlayerSystem.h"
#include "../include/LogicSystem.h"

using namespace PhysicsEngine;

System* PhysicsEngine::loadSystem(unsigned char* data)
{
	int type = *reinterpret_cast<int*>(data);

	//std::cout << "load internal system called system of type: " << type << std::endl;

	if (type == 10){
		return new LogicSystem(data);
	}
	else if (type == 11){
		return new PlayerSystem(data);
	}
	else{
		std::cout << "Error: Invalid system type (" << type << ") when trying to load internal system" << std::endl;
		return NULL;
	}
}