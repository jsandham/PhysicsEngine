#ifndef __SHADOW_MAP_DATA_H__
#define __SHADOW_MAP_DATA_H__

#include <GL/glew.h>
#include <gl/gl.h>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "../core/Shader.h"

namespace PhysicsEngine
{
	struct ShadowMapData
	{
		// directional light cascade shadow map data
		GLuint shadowCascadeFBO[5];
		GLuint shadowCascadeDepth[5];
		float cascadeEnds[6];
		glm::mat4 cascadeOrthoProj[5];
		glm::mat4 cascadeLightView[5];
		Shader depthShader;

		// spotlight shadow map data
		GLuint shadowSpotlightFBO;
		GLuint shadowSpotlightDepth;
		glm::mat4 shadowViewMatrix;
		glm::mat4 shadowProjMatrix;

		// pointlight cubemap shadow map data
		GLuint shadowCubemapFBO;
		GLuint shadowCubemapDepth;
		glm::mat4 cubeViewProjMatrices[6];
		Shader depthCubemapShader;
	};
}

#endif
