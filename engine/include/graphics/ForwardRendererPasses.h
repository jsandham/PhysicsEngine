#ifndef __FORWARD_RENDERING_PASSES_H__
#define __FORWARD_RENDERING_PASSES_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "../components/MeshRenderer.h"

#include "GraphicsQuery.h"
#include "GraphicsDebug.h"
#include "GraphicsState.h"
#include "GraphicsTargets.h"
#include "RenderObject.h"
#include "ShadowMapData.h"
#include "ScreenData.h"

namespace PhysicsEngine
{
	void initializeRenderer(World* world, ScreenData& screenData, ShadowMapData& shadowMapData, GraphicsCameraState& cameraState, GraphicsLightState& lightState, GraphicsDebug& debug, GraphicsQuery& query);
	void registerRenderAssets(World* world);
	void registerRenderObjects(World* world, std::vector<RenderObject>& renderObjects);
	void registerCameras(World* world);
	void cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects);
	void updateTransforms(World* world, std::vector<RenderObject>& renderObjects);
	void beginFrame(Camera* camera, GraphicsCameraState& cameraState, GraphicsLightState& lightState, GraphicsQuery& query);
	void computeSSAO(World* world, Camera* camera, const std::vector<RenderObject>& renderObjects, ScreenData& screenData, GraphicsQuery& query);
	void renderShadows(World* world, Camera* camera, Light* light, const std::vector<RenderObject>& renderObjects, ShadowMapData& shadowMapData, GraphicsQuery& query);
	void renderOpaques(World* world, Camera* camera, Light* light, const std::vector<RenderObject>& renderObjects, const ShadowMapData& shadowMapData, GraphicsLightState& lightState, GraphicsQuery& query);
	void renderTransparents();
	void postProcessing();
	void endFrame(World* world, Camera* camera, const std::vector<RenderObject>& renderObjects, ScreenData& screenData, GraphicsTargets& targets, GraphicsDebug& debug, GraphicsQuery& query, bool renderToScreen);

	void calcShadowmapCascades(Camera* camera, ShadowMapData& shadowMapData);
	void calcCascadeOrthoProj(Camera* camera, Light* light, ShadowMapData& shadowMapData);
	void addToRenderObjectsList(World* world, MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects);
	void removeFromRenderObjectsList(MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects);
}

#endif
