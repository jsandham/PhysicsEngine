#ifndef __FORWARD_RENDERING_PASSES_H__
#define __FORWARD_RENDERING_PASSES_H__

#include <vector>
#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/World.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "../components/MeshRenderer.h"

#include "ForwardRendererState.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
	void initializeRenderer(World* world, ForwardRendererState* state);

	void beginFrame(World* world, Camera* camera, ForwardRendererState* state);

	void computeSSAO(World* world, 
		Camera* camera, 
		ForwardRendererState* state, 
		const std::vector<RenderObject>& renderObjects);

	void renderShadows(World* world,
		Camera* camera,
		Light* light,
		Transform* lightTransform,
		ForwardRendererState* state,
		const std::vector<RenderObject>& renderObjects);

	void renderOpaques(World* world,
		Camera* camera,
		Light* light,
		Transform* lightTransform,
		ForwardRendererState* state,
		const std::vector<RenderObject>& renderObjects);

	void renderColorPicking(World* world,
		Camera* camera,
		ForwardRendererState* state,
		const std::vector<RenderObject>& renderObjects);

	void renderTransparents();

	void postProcessing();

	void endFrame(World* world, Camera* camera, ForwardRendererState* state);

	void calcShadowmapCascades(Camera* camera, ForwardRendererState* state);

	void calcCascadeOrthoProj(Camera* camera, glm::vec3 lightDirection, ForwardRendererState* state);
}

#endif
