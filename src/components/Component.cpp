#include <iostream>
#include "Component.h"

using namespace PhysicsEngine;

Component::Component()
{
	//std::cout << "component base class constructor called" << std::endl;
}

Component::~Component()
{
	//std::cout << "component base class destructor called" << std::endl;
}

//Entity* Component::getEntity()
//{
//	return entity;
//}