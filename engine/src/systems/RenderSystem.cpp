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
	mRenderToScreen = true;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	deserialize(data);
}

RenderSystem::~RenderSystem()
{
}

std::vector<char> RenderSystem::serialize() const
{
	return serialize(mSystemId);
}

std::vector<char> RenderSystem::serialize(Guid systemId) const
{
	std::vector<char> data(sizeof(int));

	memcpy(&data[0], &mOrder, sizeof(int));

	return data;
}

void RenderSystem::deserialize(std::vector<char> data)
{
	mOrder = *reinterpret_cast<int*>(&data[0]);
}

void RenderSystem::init(World* world)
{
	mWorld = world;

	mForwardRenderer.init(world, mRenderToScreen);
}

void RenderSystem::update(Input input)
{
	mForwardRenderer.update(input);
}

GraphicsTargets RenderSystem::getGraphicsTargets() const
{
	return mForwardRenderer.getGraphicsTargets();
}

GraphicsQuery RenderSystem::getGraphicsQuery() const
{
	return mForwardRenderer.getGraphicsQuery();
}