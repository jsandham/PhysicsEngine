#ifndef __TRANSFORM_GIZMOS_H__
#define __TRANSFORM_GIZMOS_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "glm/glm.hpp"
#include "glm/detail/func_trigonometric.hpp"
#include "glm/gtc/constants.hpp"

namespace PhysicsEditor
{
	typedef enum Axis
	{
		Axis_X = 0,
		Axis_Y = 1,
		Axis_Z = 2,
		Axis_None = 3
	}Axis;

	class TransformGizmo
	{
		private:
			bool mIsInitialized;

			GLuint mTranslationVAO[3];
			GLuint mTranslationVBO[3];
			GLuint mRotationVAO[3];
			GLuint mRotationVBO[3];

			std::string mVertexShader;
			std::string mFragmentShader;
			GLuint mGizmoShaderProgram;
			int mGizmoShaderMVPLoc;
			int mGizmoShaderColorLoc;

		public:
			void initialize();
			void drawTranslation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, PhysicsEngine::Ray cameraRay);
			void drawRotation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, PhysicsEngine::Ray cameraRay);
			void drawScale(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, PhysicsEngine::Ray cameraRay);
	};
}

#endif