#ifndef __GIZMOS_H__
#define __GIZMOS_H__

#include <GL/glew.h>

#include "../glm/glm.hpp"
#include "../glm/detail/func_trigonometric.hpp"
#include "../glm/gtc/constants.hpp"

#include "../components/Transform.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
	typedef enum Axis
	{
		Axis_X = 0,
		Axis_Y = 1,
		Axis_Z = 2,
		Axis_None = 3
	}Axis;

	struct GizmoState
	{
		bool mIsInitialized;

		GLuint mTranslationVAO[3];
		GLuint mTranslationVBO[3];
		GLuint mRotationVAO[3];
		GLuint mRotationVBO[3];

		Shader mGizmoShader;
		int mGizmoShaderProgram;
		int mGizmoShaderMVPLoc;
		int mGizmoShaderColorLoc;
	};

	class Gizmos
	{
		private:
			static GizmoState mState;

		public:
			static void initializeGizmos();
			static void drawTranslationGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis);
			static void drawRotationGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis);
			static void drawScaleGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis);
	};
}

#endif