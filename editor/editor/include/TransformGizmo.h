#ifndef __TRANSFORM_GIZMOS_H__
#define __TRANSFORM_GIZMOS_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "glm/glm.hpp"
#include "glm/detail/func_trigonometric.hpp"
#include "glm/gtc/constants.hpp"

#include "EditorCameraSystem.h"

namespace PhysicsEditor
{
	typedef enum Axis
	{
		Axis_X = 0,
		Axis_Y = 1,
		Axis_Z = 2,
		Axis_None = 3
	}Axis;

	typedef enum GizmoMode
	{
		Translation = 0,
		Rotation = 1,
		Scale = 2
	};

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

			GizmoMode mode;
			Axis highlightedTransformAxis;
			Axis selectedTransformAxis;
			glm::mat4 selectedTransformModel;

		public:
			void initialize();
			void update(PhysicsEngine::EditorCameraSystem * cameraSystem, PhysicsEngine::Transform* selectedTransform, float contentWidth, float contentHeight);
			void setGizmoMode(GizmoMode mode);

			bool isGizmoHighlighted() const;

		private:
			void drawTranslation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, Axis highlightAxis, Axis selectedAxis);
			void drawRotation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, Axis highlightAxis, Axis selectedAxis);
			void drawScale(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, Axis highlightAxis, Axis selectedAxis);
	};
}

#endif