#include <iostream>

//#include <components/Transform.h>
#include <core/Guid.h>

using namespace PhysicsEngine;

int main()
{
	std::cout << "Hello world" << std::endl;

	//Transform transform;
	//std::cout << "position: " << transform.mPosition.x << std::endl;

	Guid id = Guid::newGuid();
	//std::cout << "id: " << id.toString() << std::endl;

	int i = 0;
	while (i < 10000)
	{
		std::cout << "i: " << i << std::endl;
		i++;
	}

	return 0;
}