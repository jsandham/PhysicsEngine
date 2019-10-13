#include <iostream>
#include <cstddef>
#include <ctime>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/ForwardRenderer.h"

#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	pass = 0;
	renderToScreen = true;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	deserialize(data);
}

RenderSystem::~RenderSystem()
{
}

std::vector<char> RenderSystem::serialize()
{
	size_t numberOfBytes = sizeof(int);
	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &order, sizeof(int));

	return data;
}

void RenderSystem::deserialize(std::vector<char> data)
{
	order = *reinterpret_cast<int*>(&data[0]);
}

void RenderSystem::init(World* world)
{
	this->world = world;

	forwardRenderer.init(world, renderToScreen);
}

void RenderSystem::update(Input input)
{
	forwardRenderer.update(input);
}

GraphicsTargets RenderSystem::getGraphicsTargets() const
{
	return forwardRenderer.getGraphicsTargets();
}

GraphicsQuery RenderSystem::getGraphicsQuery() const
{
	return forwardRenderer.getGraphicsQuery();
}