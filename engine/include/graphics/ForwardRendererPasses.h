#ifndef __FORWARD_RENDERING_PASSES_H__
#define __FORWARD_RENDERING_PASSES_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "GraphicsQuery.h"
#include "GraphicsDebug.h"
#include "GraphicsState.h"
#include "GraphicsTargets.h"
#include "RenderObject.h"
#include "ShadowMapData.h"
#include "FramebufferData.h"

namespace PhysicsEngine
{
	void initializeForwardRenderer(World* world, FramebufferData& fboData, ShadowMapData& shadowMapData, GraphicsCameraState& cameraState, GraphicsLightState& lightState, GraphicsDebug& debug, GraphicsQuery& query);
	void registerRenderAssets(World* world);
	void registerRenderObjects(World* world, std::vector<RenderObject>& renderObjects);
	void cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects);
	void updateTransforms(World* world, std::vector<RenderObject>& renderObjects);
	void beginFrame(Camera* camera, FramebufferData& fboData, GraphicsCameraState& cameraState, GraphicsQuery& query);
	void renderShadows(World* world, Camera* camera, Light* light, const std::vector<RenderObject>& renderObjects, ShadowMapData& shadowMapData, GraphicsQuery& query);
	void renderOpaques(World* world, Camera* camera, Light* light, FramebufferData& fboData, const ShadowMapData& shadowMapData, GraphicsLightState& lightState, const std::vector<RenderObject>& renderObjects, GraphicsQuery& query);
	void renderTransparents();
	void postProcessing();
	void endFrame(World* world, const std::vector<RenderObject>& renderObjects, FramebufferData& fboData, GraphicsTargets& targets, GraphicsDebug& debug, GraphicsQuery& query, bool renderToScreen);



	void calcShadowmapCascades(Camera* camera, ShadowMapData& shadowMapData);
	void calcCascadeOrthoProj(Camera* camera, Light* light, ShadowMapData& shadowMapData);
	void addToRenderObjectsList(World* world, MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects);
	void removeFromRenderObjectsList(MeshRenderer* meshRenderer, std::vector<RenderObject>& renderObjects);
}

#endif
