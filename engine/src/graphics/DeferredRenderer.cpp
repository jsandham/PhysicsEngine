#include"../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/DeferredRendererPasses.h"

using namespace PhysicsEngine;

DeferredRenderer::DeferredRenderer()
{

}

DeferredRenderer::~DeferredRenderer()
{

}

void DeferredRenderer::init(World* world, bool renderToScreen)
{
	mWorld = world;
	mState.mRenderToScreen = renderToScreen;

	initializeDeferredRenderer(mWorld, &mState);
}

void DeferredRenderer::update(Input input, Camera* camera, std::vector<RenderObject>& renderObjects)
{
	beginDeferredFrame(mWorld, camera, &mState);

	geometryPass(&mState, renderObjects);
	lightingPass(&mState, renderObjects);
	
	endDeferredFrame(mWorld, camera, &mState);



	//Camera* camera;
	//if(world->getNumberOfComponents<Camera>() > 0){
	//	camera = world->getComponentByIndex<Camera>(0);
	//}
	//else{
	//	std::cout << "Warning: No camera found" << std::endl;
	//	return;
	//}

	//glm::mat4 projection = camera->getProjMatrix();
	//glm::mat4 view = camera->getViewMatrix();

	//// geometry pass
	//glBindFramebuffer(GL_FRAMEBUFFER, gbuffer.handle);

	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Graphics::use(&gbuffer.shader, ShaderVariant::None);
	//gbuffer.shader.setMat4("projection", ShaderVariant::None, projection);
	//gbuffer.shader.setMat4("view", ShaderVariant::None, view);

	//std::cout << "number of render objects: " << renderObjects.size() << std::endl;

	//for(size_t i = 0; i < renderObjects.size(); i++){
	//	Transform* transform = world->getComponentByIndex<Transform>(renderObjects[i].transformIndex);

	//	renderObjects[i].model = transform->getModelMatrix();
	//}

	//glBindVertexArray(meshBuffer.vao);

	//for(size_t i = 0; i < renderObjects.size(); i++){
	//	Material* material = world->getAssetByIndex<Material>(renderObjects[i].materialIndex);

	//	gbuffer.shader.setMat4("model", ShaderVariant::None, renderObjects[i].model);

	//	GLsizei numVertices = renderObjects[i].size / 3;
	//	GLint startIndex = renderObjects[i].start / 3;

	//	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);
	//}

	//glBindVertexArray(0);

	//// if(getKeyDown(input, KeyCode::X)){
	//// 	std::cout << "XXXXXXXXXXXXXX" << std::endl;
	//// 	data.resize(4 * camera->width * camera->height);

	//// 	//glReadBuffer(GL_DEPTH_COMPONENT);
	//// 	//glReadPixels(0, 0, camera->width, camera->height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, &data[0]);
	//// 	glReadBuffer(GL_COLOR_ATTACHMENT2);
	//// 	glReadPixels(0, 0, camera->width, camera->height, GL_RGBA, GL_UNSIGNED_BYTE, &data[0]);


	//// 	std::cout << "Read successfull" << std::endl;

	//// 	World::writeToBMP("deferred.bmp", data, camera->width, camera->height, 4);
	//// }

	//glBindFramebuffer(GL_FRAMEBUFFER, 0);


	//// lighting pass
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glActiveTexture(GL_TEXTURE0);
 //   glBindTexture(GL_TEXTURE_2D, gbuffer.color0);
 //   glActiveTexture(GL_TEXTURE1);
 //   glBindTexture(GL_TEXTURE_2D, gbuffer.color1);
 //   glActiveTexture(GL_TEXTURE2);
 //   glBindTexture(GL_TEXTURE_2D, gbuffer.color2);



 //   Graphics::checkError();
	//// GLenum error;
	//// while ((error = glGetError()) != GL_NO_ERROR){
	//// 	std::cout << "Error: Deferred Renderer failed with error code: " << error << " during update" << std::endl;;
	//// }
}