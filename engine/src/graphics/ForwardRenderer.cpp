#include "../../include/graphics/ForwardRenderer.h"
#include "../../include/graphics/ForwardRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"
#include "../../include/core/Util.h"

using namespace PhysicsEngine;

ForwardRenderer::ForwardRenderer()
{
	
}

ForwardRenderer::~ForwardRenderer()
{

}

void ForwardRenderer::init(World* world, bool renderToScreen)
{
	this->world = world;
	this->renderToScreen = renderToScreen;

	initializeRenderer(world, screenData, shadowMapData, cameraState, lightState, debug, query);
}

void ForwardRenderer::update(Input input)
{
	registerRenderAssets(world);
	registerRenderObjects(world, renderObjects);
	registerCameras(world);

	updateTransforms(world, renderObjects);

	for (int i = 0; i < world->getNumberOfComponents<Camera>(); i++) {
		Camera* camera = world->getComponentByIndex<Camera>(i);

		cullRenderObjects(camera, renderObjects);

		beginFrame(camera, cameraState, query);

		computeSSAO(world, camera, renderObjects, screenData, query);

		for (int j = 0; j < world->getNumberOfComponents<Light>(); j++) {
			Light* light = world->getComponentByIndex<Light>(j);

			renderShadows(world, camera, light, renderObjects, shadowMapData, query);
			renderOpaques(world, camera, light, renderObjects, shadowMapData, lightState, query);
			renderTransparents();
		}

		postProcessing();
		endFrame(world, camera, renderObjects, screenData, targets, debug, query, renderToScreen);
	}
}

GraphicsQuery ForwardRenderer::getGraphicsQuery() const
{
	return query;
}

GraphicsDebug ForwardRenderer::getGraphicsDebug() const
{
	return debug;
}

GraphicsTargets ForwardRenderer::getGraphicsTargets() const
{
	return targets;
}