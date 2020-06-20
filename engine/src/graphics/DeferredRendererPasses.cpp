#include "../../include/graphics/DeferredRendererPasses.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/core/Shader.h"
#include "../../include/core/InternalShaders.h"

using namespace PhysicsEngine;

void PhysicsEngine::initializeDeferredRenderer(World* world, DeferredRendererState* state)
{
	// generate all internal shader programs
	state->mGeometryShader.setVertexShader(InternalShaders::gbufferVertexShader);
	state->mGeometryShader.setFragmentShader(InternalShaders::gbufferFragmentShader);
	state->mGeometryShader.compile();

	// cache internal shader uniforms
	state->mGeometryShaderProgram = state->mGeometryShader.getProgramFromVariant(ShaderVariant::None);
	state->mGeometryShaderModelLoc = state->mGeometryShader.findUniformLocation("model", state->mGeometryShaderProgram);
	state->mGeometryShaderDiffuseTexLoc = state->mGeometryShader.findUniformLocation("texture_diffuse1", state->mGeometryShaderProgram);
	state->mGeometryShaderSpecTexLoc = state->mGeometryShader.findUniformLocation("texture_specular1", state->mGeometryShaderProgram);

	//generate screen quad for final rendering
	constexpr float quadVertices[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};

	glGenVertexArrays(1, &state->mQuadVAO);
	glBindVertexArray(state->mQuadVAO);

	glGenBuffers(1, &state->mQuadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, state->mQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	Graphics::checkError();

	glGenBuffers(1, &(state->mCameraState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &(state->mLightState.mHandle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
	glBufferData(GL_UNIFORM_BUFFER, 824, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	Graphics::checkError();
}

void PhysicsEngine::beginDeferredFrame(World* world, Camera* camera, DeferredRendererState* state)
{
	camera->mQuery.mNumBatchDrawCalls = 0;
	camera->mQuery.mNumDrawCalls = 0;
	camera->mQuery.mTotalElapsedTime = 0.0f;
	camera->mQuery.mVerts = 0;
	camera->mQuery.mTris = 0;
	camera->mQuery.mLines = 0;
	camera->mQuery.mPoints = 0;

	glBeginQuery(GL_TIME_ELAPSED, camera->mQuery.mQueryId[camera->mQuery.mQueryBack]);

	state->mCameraState.mProjection = camera->getProjMatrix();
	state->mCameraState.mView = camera->getViewMatrix();
	state->mCameraState.mCameraPos = camera->getComponent<Transform>(world)->mPosition;

	// set camera state binding point and update camera state data
	glBindBuffer(GL_UNIFORM_BUFFER, state->mCameraState.mHandle);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->mCameraState.mHandle, 0, 144);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state->mCameraState.mProjection));
	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state->mCameraState.mView));
	glBufferSubData(GL_UNIFORM_BUFFER, 128, 12, glm::value_ptr(state->mCameraState.mCameraPos));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	// set light state binding point
	glBindBuffer(GL_UNIFORM_BUFFER, state->mLightState.mHandle);
	glBindBufferRange(GL_UNIFORM_BUFFER, 1, state->mLightState.mHandle, 0, 824);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glClearColor(camera->mBackgroundColor.x, camera->mBackgroundColor.y, camera->mBackgroundColor.z, camera->mBackgroundColor.w);
	glClearDepth(1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsMainFBO());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsGeometryFBO());
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PhysicsEngine::geometryPass(DeferredRendererState* state, const std::vector<RenderObject>& renderObjects)
{

}

void PhysicsEngine::lightingPass(DeferredRendererState* state, const std::vector<RenderObject>& renderObjects)
{

}

void PhysicsEngine::endDeferredFrame(World* world, Camera* camera, DeferredRendererState* state)
{
	glEndQuery(GL_TIME_ELAPSED);

	GLuint64 elapsedTime; // in nanoseconds
	glGetQueryObjectui64v(camera->mQuery.mQueryId[camera->mQuery.mQueryFront], GL_QUERY_RESULT, &elapsedTime);

	camera->mQuery.mTotalElapsedTime += elapsedTime / 1000000.0f;

	// swap which query is active
	if (camera->mQuery.mQueryBack) {
		camera->mQuery.mQueryBack = 0;
		camera->mQuery.mQueryFront = 1;
	}
	else {
		camera->mQuery.mQueryBack = 1;
		camera->mQuery.mQueryFront = 0;
	}

	if (state->mRenderToScreen) {
		glViewport(0, 0, 1024, 1024);
		glScissor(0, 0, 1024, 1024);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		state->mQuadShader.use(ShaderVariant::None);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorTex());

		glBindVertexArray(state->mQuadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}
}