#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/Gizmos.h"
#include "../../include/core/InternalShaders.h"

using namespace PhysicsEngine;

GizmoState Gizmos::mState;

void Gizmos::initializeGizmos()
{
	mState.mGizmoShader.setVertexShader(InternalShaders::gizmoVertexShader);
	mState.mGizmoShader.setFragmentShader(InternalShaders::gizmoFragmentShader);
	mState.mGizmoShader.compile();

	mState.mGizmoShaderProgram = mState.mGizmoShader.getProgramFromVariant(PhysicsEngine::ShaderVariant::None);
	mState.mGizmoShaderMVPLoc = mState.mGizmoShader.findUniformLocation("mvp", mState.mGizmoShaderProgram);
	mState.mGizmoShaderColorLoc = mState.mGizmoShader.findUniformLocation("color", mState.mGizmoShaderProgram);

	for (int i = 0; i < 3; i++) {
		GLfloat translationVertices[] =
		{
			0.0f, 0.0f, 0.0f, // first vertex
			1.0f * (i == 0), 1.0f * (i == 1), 1.0f * (i == 2)  // second vertex
		};

		glGenVertexArrays(1, &mState.mTranslationVAO[i]);
		glBindVertexArray(mState.mTranslationVAO[i]);

		glGenBuffers(1, &mState.mTranslationVBO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, mState.mTranslationVBO[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(translationVertices), &translationVertices[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		Graphics::checkError();
	}

	constexpr int NUM_VERTS = 60;

	GLfloat rotationVertices[3 * 3 * 120];
	for (int i = 0; i < 60; i++) {
		float s1 = glm::cos(2 * glm::pi<float>() * i / 60.0f);
		float t1 = glm::sin(2 * glm::pi<float>() * i / 60.0f);
		float s2 = glm::cos(2 * glm::pi<float>() * (i + 1) / 60.0f);
		float t2 = glm::sin(2 * glm::pi<float>() * (i + 1) / 60.0f);

		rotationVertices[6 * i + 0] = 0.0f;
		rotationVertices[6 * i + 1] = s1;
		rotationVertices[6 * i + 2] = t1;
		rotationVertices[6 * i + 3] = 0.0f;
		rotationVertices[6 * i + 4] = s2;
		rotationVertices[6 * i + 5] = t2;

		rotationVertices[6 * i + 0 + 3 * 120] = s1;
		rotationVertices[6 * i + 1 + 3 * 120] = 0.0f;
		rotationVertices[6 * i + 2 + 3 * 120] = t1;
		rotationVertices[6 * i + 3 + 3 * 120] = s2;
		rotationVertices[6 * i + 4 + 3 * 120] = 0.0f;
		rotationVertices[6 * i + 5 + 3 * 120] = t2;

		rotationVertices[6 * i + 0 + 6 * 120] = s1;
		rotationVertices[6 * i + 1 + 6 * 120] = t1;
		rotationVertices[6 * i + 2 + 6 * 120] = 0.0f;
		rotationVertices[6 * i + 3 + 6 * 120] = s2;
		rotationVertices[6 * i + 4 + 6 * 120] = t2;
		rotationVertices[6 * i + 5 + 6 * 120] = 0.0f;
	}

	for (int i = 0; i < 3; i++)
	{
		glGenVertexArrays(1, &mState.mRotationVAO[i]);
		glBindVertexArray(mState.mRotationVAO[i]);

		glGenBuffers(1, &mState.mRotationVBO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, mState.mRotationVBO[i]);
		glBufferData(GL_ARRAY_BUFFER, 3 * 120 * sizeof(GLfloat), &rotationVertices[i * 3 * 120], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	glEnable(GL_LINE_SMOOTH);

	Graphics::checkError();

	mState.mIsInitialized = true;
}

void Gizmos::drawTranslationGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis)
{
	if (!mState.mIsInitialized) {
		initializeGizmos();
	}

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glClear(GL_DEPTH_BUFFER_BIT);

	for (int i = 0; i < 3; i++)
	{
		glm::vec4 axis_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		axis_color[i] = 1.0f;

		if (i == static_cast<int>(selectedAxis))
		{
			axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
		}

		glm::mat4 mvp = projection * view * transform->getModelMatrix();

		mState.mGizmoShader.use(mState.mGizmoShaderProgram);
		mState.mGizmoShader.setMat4(mState.mGizmoShaderMVPLoc, mvp);
		mState.mGizmoShader.setVec4(mState.mGizmoShaderColorLoc, axis_color);
		glBindVertexArray(mState.mTranslationVAO[i]);
		glLineWidth(2.0f);
		glDrawArrays(GL_LINES, 0, 2);
		glBindVertexArray(0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Gizmos::drawRotationGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis)
{
	if (!mState.mIsInitialized) {
		initializeGizmos();
	}

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glClear(GL_DEPTH_BUFFER_BIT);

	for (int i = 0; i < 3; i++)
	{
		glm::vec4 axis_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		axis_color[i] = 1.0f;

		if (i == static_cast<int>(selectedAxis))
		{
			axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
		}

		glm::mat4 mvp = projection * view * transform->getModelMatrix();

		mState.mGizmoShader.use(mState.mGizmoShaderProgram);
		mState.mGizmoShader.setMat4(mState.mGizmoShaderMVPLoc, mvp);
		mState.mGizmoShader.setVec4(mState.mGizmoShaderColorLoc, axis_color);
		glBindVertexArray(mState.mRotationVAO[i]);
		glLineWidth(2.0f);
		glDrawArrays(GL_LINES, 0, 120);
		glBindVertexArray(0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Gizmos::drawScaleGizmo(Transform* transform, glm::mat4 projection, glm::mat4 view, GLuint fbo, Axis selectedAxis)
{
	if (!mState.mIsInitialized) {
		initializeGizmos();
	}

}