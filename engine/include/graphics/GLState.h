#ifndef __GRAPHICSTATE_H__
#define __GRAPHICSTATE_H__

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	typedef enum UniformBuffer
	{
		CameraBuffer,
		ShadowBuffer,
		DirectionalLightBuffer,
		SpotLightBuffer,
		PointLightBuffer
	}UniformBuffer;

	struct GLCamera
	{
		glm::mat4 projection;         // 0
		glm::mat4 view;               // 64
		glm::vec3 cameraPos;          // 128

		GLHandle handle;
	};

	struct GLShadow
	{
		glm::mat4 lightProjection[5]; // 0    64   128  192  256
		glm::mat4 lightView[5];       // 320  384  448  512  576 
		float cascadeEnd[5];          // 640  656  672  688  704
		float farPlane;               // 720

		GLHandle handle;
	};

	struct GLDirectionalLight
	{
		glm::vec3 dirLightDirection;  // 0
		glm::vec3 dirLightAmbient;    // 16
		glm::vec3 dirLightDiffuse;    // 32
		glm::vec3 dirLightSpecular;   // 48

		GLHandle handle;
	};

	struct GLSpotLight
	{
		glm::vec3 spotLightPosition;  // 0
		glm::vec3 spotLightDirection; // 16
		glm::vec3 spotLightAmbient;   // 32
		glm::vec3 spotLightDiffuse;   // 48
		glm::vec3 spotLightSpecular;  // 64
		float spotLightConstant;      // 80
		float spotLightLinear;        // 84
		float spotLightQuadratic;     // 88
		float spotLightCutoff;        // 92
		float spotLightOuterCutoff;   // 96

		GLHandle handle;
	};

	struct GLPointLight
	{
		glm::vec3 pointLightPosition; // 0
		glm::vec3 pointLightAmbient;  // 16
		glm::vec3 pointLightDiffuse;  // 32
		glm::vec3 pointLightSpecular; // 48
		float pointLightConstant;     // 64
		float pointLightLinear;       // 68
		float pointLightQuadratic;    // 72

		GLHandle handle;
	};
}

#endif