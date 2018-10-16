#include <cstddef>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/Graphics.h"

#include "../../include/core/Manager.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	type = 0;
	numLights = 0;
}

RenderSystem::RenderSystem(unsigned char* data)
{
	type = 0;
	numLights = 0;
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::init()
{
	std::cout << "render system init called" << std::endl;

	for(int i = 0; i < manager->getNumberOfTextures(); i++){
		Graphics::generate(manager->getTexture2DByIndex(i));
	}

	for(int i = 0; i < manager->getNumberOfShaders(); i++){
		Shader* shader = manager->getShaderByIndex(i);

		shader->compile();

		if(!shader->isCompiled()){
			std::cout << "Shader failed to compile " << i << std::endl;
		}

		std::cout << "shader: " << shader->shaderId << std::endl;

		Graphics::setUniformBlockToBindingPoint(shader, "CameraBlock", 0);
		Graphics::setUniformBlockToBindingPoint(shader, "DirectionalLightBlock", 2);
		Graphics::setUniformBlockToBindingPoint(shader, "SpotLightBlock", 3);
		Graphics::setUniformBlockToBindingPoint(shader, "PointLightBlock", 4);
	}

	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
	for(int i = 0; i < manager->getNumberOfMeshes(); i++){
		Mesh* mesh = manager->getMeshByIndex(i);

		Graphics::generate(mesh);
	}
	
	Graphics::generate(&cameraState);
	Graphics::generate(&directionLightState);
	Graphics::generate(&spotLightState);
	Graphics::generate(&pointLightState);

	Graphics::enableDepthTest();
	Graphics::enableCubemaps();
	Graphics::enablePoints();
	Graphics::checkError();

	std::cout << "Render system init called" << std::endl;
	std::cout << "GL_TEXTURE_2D: " << GL_TEXTURE_2D << std::endl;
}

void RenderSystem::update()
{
	int numberOfDirectionalLights = manager->getNumberOfDirectionalLights();
	int numberOfSpotLights = manager->getNumberOfSpotLights();
	int numberOfPointLights = manager->getNumberOfPointLights();

	Camera* camera;
	if(manager->getNumberOfCameras() > 0){
		camera = manager->getCameraByIndex(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	Graphics::setViewport(camera->x, camera->y, camera->width, camera->height);
	Graphics::clearColorBuffer(camera->getBackgroundColor());
	Graphics::clearDepthBuffer(1.0f);

	projection = camera->getProjMatrix();
	view = camera->getViewMatrix();
	cameraPos = camera->getPosition();

	Graphics::bind(&cameraState);
	Graphics::setProjectionMatrix(&cameraState, camera->getProjMatrix());
	Graphics::setViewMatrix(&cameraState, camera->getViewMatrix());
	Graphics::setCameraPosition(&cameraState, camera->getPosition());
	Graphics::unbind(&cameraState);

	pass = 0;

	if (numberOfDirectionalLights > 0){
		DirectionalLight* directionalLight = manager->getDirectionalLightByIndex(0);

		Graphics::bind(&directionLightState);
		Graphics::setDirLightDirection(&directionLightState, -directionalLight->direction);
		Graphics::setDirLightAmbient(&directionLightState, directionalLight->ambient);
		Graphics::setDirLightDiffuse(&directionLightState, directionalLight->diffuse);
		Graphics::setDirLightSpecular(&directionLightState, directionalLight->specular);
		Graphics::unbind(&directionLightState);

		renderScene();

		pass++;
	}

	for(int i = 0; i < numberOfSpotLights; i++){
		SpotLight* spotLight = manager->getSpotLightByIndex(i);

		glm::vec3 position = spotLight->position;
		glm::vec3 direction = spotLight->direction;
		glm::mat4 lightProjection = spotLight->projection;
		glm::mat4 lightView = glm::lookAt(position, position - direction, glm::vec3(0.0f, 1.0f, 0.0f));

		Graphics::bind(&spotLightState);
		Graphics::setSpotLightPosition(&spotLightState, spotLight->position);
		Graphics::setSpotLightDirection(&spotLightState, spotLight->direction);
		Graphics::setSpotLightAmbient(&spotLightState, spotLight->ambient);
		Graphics::setSpotLightDiffuse(&spotLightState, spotLight->diffuse);
		Graphics::setSpotLightSpecular(&spotLightState, spotLight->specular);
		Graphics::setSpotLightConstant(&spotLightState, spotLight->constant);
		Graphics::setSpotLightLinear(&spotLightState, spotLight->linear);
		Graphics::setSpotLightQuadratic(&spotLightState, spotLight->quadratic);
		Graphics::setSpotLightCutoff(&spotLightState, spotLight->cutOff);
		Graphics::setSpotLightOuterCutoff(&spotLightState, spotLight->outerCutOff);
		Graphics::unbind(&spotLightState);

		renderScene();

		pass++;
	}

	for(int i = 0; i < numberOfPointLights; i++){
		PointLight* pointLight = manager->getPointLightByIndex(i);

		glm::vec3 lightPosition = pointLight->position;
		glm::mat4 lightProjection = pointLight->projection;

		Graphics::bind(&pointLightState);
		Graphics::setPointLightPosition(&pointLightState, pointLight->position);
		Graphics::setPointLightAmbient(&pointLightState, pointLight->ambient);
		Graphics::setPointLightDiffuse(&pointLightState, pointLight->diffuse);
		Graphics::setPointLightSpecular(&pointLightState, pointLight->specular);
		Graphics::setPointLightConstant(&pointLightState, pointLight->constant);
		Graphics::setPointLightLinear(&pointLightState, pointLight->linear);
		Graphics::setPointLightQuadratic(&pointLightState, pointLight->quadratic);
		Graphics::unbind(&pointLightState);

		renderScene();
			
		pass++;
	}

	Graphics::checkError();
}

void RenderSystem::renderScene()
{
	for(int i = 0; i < manager->getNumberOfMeshRenderers(); i++){
		MeshRenderer* meshRenderer = manager->getMeshRendererByIndex(i);
		Transform* transform = meshRenderer->getComponent<Transform>();

		//std::cout << "entity id: " << meshRenderer->entityId << " mesh id: " << meshRenderer->meshId << " material id: " << meshRenderer->materialId << std::endl;

		Mesh* mesh = manager->getMesh(meshRenderer->meshId);
		Material* material = manager->getMaterial(meshRenderer->materialId);  // should I call somethig like Graphics::bind(material) and then have this internally use the shader and bind any textures?
		//Shader* shader = manager->getShader(material->shaderId);
		//Texture2D* texture = manager->getTexture2D(material->textureId);

		glm::mat4 model = transform->getModelMatrix();

		// std::cout << "shader compiled: " << shader->isCompiled() << " id: " << shader->shaderId << std::endl;
		// std::cout << view[0][0] << " " << view[0][1] << " " << view[0][2] << " " << view[0][3] << std::endl;
		// std::cout << view[1][0] << " " << view[1][1] << " " << view[1][2] << " " << view[1][3] << std::endl;
		// std::cout << view[2][0] << " " << view[2][1] << " " << view[2][2] << " " << view[2][3] << std::endl;
		// std::cout << view[3][0] << " " << view[3][1] << " " << view[3][2] << " " << view[3][3] << std::endl;
		// glm::vec3 position = transform->position;
		// glm::quat rotation = transform->rotation;
		// glm::vec3 scale = transform->scale;

		// std::cout << "transform i: " << i << std::endl;
		// std::cout <<"position: " << position.x << " " << position.y << " " << position.z << std::endl;
		// std::cout <<"rotation: " << rotation.x << " " << rotation.y << " " << rotation.z << " " << rotation.w << std::endl;
		// std::cout <<"scale: " << scale.x << " " << scale.y << " " << scale.z << std::endl;

		// Graphics::use(shader);
		// Graphics::setMat4(shader, "model", model);

		// Graphics::bind(texture);

		Graphics::bind(material, model);
		Graphics::bind(mesh);
		Graphics::draw(mesh);
		Graphics::unbind(mesh);
	}

	Graphics::checkError();
}