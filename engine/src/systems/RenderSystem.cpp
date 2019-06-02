#include <iostream>
#include <cstddef>
#include <ctime>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/GraphicsQuery.h"

#include "../../include/core/Input.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	type = 0;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	size_t index = sizeof(char);
	type = *reinterpret_cast<int*>(&data[index]);
	index += sizeof(int);
	order = *reinterpret_cast<int*>(&data[index]);

	if(type != 0){
		std::cout << "Error: System type (" << type << ") found in data array is invalid" << std::endl;
	}
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::init(World* world)
{
	std::cout << "Render System init called" << std::endl;
	this->world = world;

	forwardRenderer.init(world);
	//deferredRenderer.init(world);
	debugRenderer.init(world);
}

void RenderSystem::update(Input input)
{
	//deferredRenderer.update(input);
	forwardRenderer.update(input);

	if(world->debug){
		GraphicsQuery query = forwardRenderer.getGraphicsQuery();
		GraphicsDebug debug = forwardRenderer.getGraphicsDebug();

		debugRenderer.update(input, debug, query);
	}
}