#include "../include/MaterialRenderer.h"

#include "core/InternalShaders.h"
#include "core/Log.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

using namespace PhysicsEditor;

MaterialRenderer::MaterialRenderer()
{

}

MaterialRenderer::~MaterialRenderer()
{

}

void MaterialRenderer::init()
{
	// create framebuffer (color + depth)
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glGenTextures(1, &colorTex);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, &depthTex);
	glBindTexture(GL_TEXTURE_2D, depthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int mainAttachments[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, mainAttachments);

	//Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//mesh.load();

	// create mesh vao and vbo
	glGenVertexArrays(1, &mesh.vao);
	glBindVertexArray(mesh.vao);
	glGenBuffers(1, &mesh.vbo[0]);
	glGenBuffers(1, &mesh.vbo[1]);
	glGenBuffers(1, &mesh.vbo[2]);

	glBindVertexArray(mesh.vao);
	glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, mesh.getVertices().size() * sizeof(float), &(mesh.getVertices()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, mesh.getNormals().size() * sizeof(float), &(mesh.getNormals()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo[2]);
	glBufferData(GL_ARRAY_BUFFER, mesh.getTexCoords().size() * sizeof(float), &(mesh.getTexCoords()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);

	shader.setVertexShader(PhysicsEngine::InternalShaders::simpleLitVertexShader);
	shader.setFragmentShader(PhysicsEngine::InternalShaders::simpleLitFragmentShader);
	shader.compile();

	if (!shader.isCompiled()) {
		std::string errorMessage = "Shader failed to compile " + shader.assetId.toString() + "\n";
		PhysicsEngine::Log::error(&errorMessage[0]);
	}

	// define mesh orientation
	glm::vec3 meshPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::quat meshRotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	glm::vec3 meshScale = glm::vec3(1.0f, 1.0f, 1.0f);

	glm::mat4 model = glm::translate(glm::mat4(), meshPosition);
	model *= glm::toMat4(meshRotation);
	model = glm::scale(model, meshScale);

	// define camera orientation
	cameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

	glm::mat4 view = glm::lookAt(cameraPos, cameraPos + front, up);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 250.0f);
}

void MaterialRenderer::render(PhysicsEngine::Material* material)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	shader.use(0);
	shader.setMat4("model", model);
	shader.setMat4("view", view);
	shader.setMat4("projection", projection);
	shader.setVec3("cameraPos", cameraPos);

	glBindVertexArray(mesh.vao);
	glDrawArrays(GL_TRIANGLES, 0, mesh.getVertices().size() / 3);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}