#ifndef __GRAPHICSTATE_H__
#define __GRAPHICSTATE_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../glm/glm.hpp"
#include "../glm/gtc/type_ptr.hpp"

namespace PhysicsEngine
{
	struct GraphicsCameraState
	{
		glm::mat4 projection;         // 0
		glm::mat4 view;               // 64
		glm::vec3 cameraPos;          // 128

		GLuint handle;
	};

	struct GraphicsShadowState
	{
		glm::mat4 lightProjection[5]; // 0    64   128  192  256
		glm::mat4 lightView[5];       // 320  384  448  512  576 
		float cascadeEnd[5];          // 640  656  672  688  704
		float farPlane;               // 720

		GLuint handle;
	};

	struct GraphicsDirectionalLightState
	{
		glm::vec3 direction;  // 0
		glm::vec3 ambient;    // 16
		glm::vec3 diffuse;    // 32
		glm::vec3 specular;   // 48

		GLuint handle;
	};

	struct GraphicsSpotLightState
	{
		glm::vec3 position;  // 0
		glm::vec3 direction; // 16
		glm::vec3 ambient;   // 32
		glm::vec3 diffuse;   // 48
		glm::vec3 specular;  // 64
		float constant;      // 80
		float linear;        // 84
		float quadratic;     // 88
		float cutOff;        // 92
		float outerCutOff;   // 96

		GLuint handle;
	};

	struct GraphicsPointLightState
	{
		glm::vec3 position; // 0
		glm::vec3 ambient;  // 16
		glm::vec3 diffuse;  // 32
		glm::vec3 specular; // 48
		float constant;     // 64
		float linear;       // 68
		float quadratic;    // 72

		GLuint handle;
	};
}

#endif