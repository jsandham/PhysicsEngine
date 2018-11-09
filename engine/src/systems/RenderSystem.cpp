#include <cstddef>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Manager.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Line.h"
#include "../../include/core/Input.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	type = 0;
}

RenderSystem::RenderSystem(unsigned char* data)
{
	type = 0;
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::init()
{
	for(int i = 0; i < manager->getNumberOfAssets<Texture2D>(); i++){
		Graphics::generate(manager->getAssetByIndex<Texture2D>(i));
	}

	for(int i = 0; i < manager->getNumberOfAssets<Shader>(); i++){
		Shader* shader = manager->getAssetByIndex<Shader>(i);

		shader->compile();

		if(!shader->isCompiled()){
			std::cout << "Shader failed to compile " << i << std::endl;
		}

		Graphics::setUniformBlockToBindingPoint(shader, "CameraBlock", 0);
		Graphics::setUniformBlockToBindingPoint(shader, "DirectionalLightBlock", 2);
		Graphics::setUniformBlockToBindingPoint(shader, "SpotLightBlock", 3);
		Graphics::setUniformBlockToBindingPoint(shader, "PointLightBlock", 4);
	}

	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
	for(int i = 0; i < manager->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = manager->getAssetByIndex<Mesh>(i);

		Graphics::generate(mesh);
	}

	Line* line = manager->getLine();

	Graphics::generate(line);
	
	Graphics::generate(&cameraState);
	Graphics::generate(&directionLightState);
	Graphics::generate(&spotLightState);
	Graphics::generate(&pointLightState);

	Graphics::enableBlend();
	Graphics::enableDepthTest();
	Graphics::enableCubemaps();
	Graphics::enablePoints();

	Graphics::checkError();
}

void RenderSystem::update()
{
	int numberOfDirectionalLights = manager->getNumberOfComponents<DirectionalLight>();
	int numberOfSpotLights = manager->getNumberOfComponents<SpotLight>();
	int numberOfPointLights = manager->getNumberOfComponents<PointLight>();

	Camera* camera;
	if(manager->getNumberOfComponents<Camera>() > 0){
		camera = manager->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	Graphics::setViewport(camera->x, camera->y, camera->width, camera->height);
	Graphics::clearColorBuffer(camera->getBackgroundColor());
	Graphics::clearDepthBuffer(1.0f);
	Graphics::setDepth(GLDepth::LEqual);
	Graphics::setBlending(GLBlend::One, GLBlend::Zero);

	Graphics::bind(&cameraState);
	Graphics::setProjectionMatrix(&cameraState, camera->getProjMatrix());
	Graphics::setViewMatrix(&cameraState, camera->getViewMatrix());
	Graphics::setCameraPosition(&cameraState, camera->getPosition());
	Graphics::unbind(&cameraState);

	pass = 0;

	if (numberOfDirectionalLights > 0){
		DirectionalLight* directionalLight = manager->getComponentByIndex<DirectionalLight>(0);

		Graphics::bind(&directionLightState);
		Graphics::setDirLightDirection(&directionLightState, directionalLight->direction);
		Graphics::setDirLightAmbient(&directionLightState, directionalLight->ambient);
		Graphics::setDirLightDiffuse(&directionLightState, directionalLight->diffuse);
		Graphics::setDirLightSpecular(&directionLightState, directionalLight->specular);
		Graphics::unbind(&directionLightState);

		renderScene();

		pass++;
	}

	for(int i = 0; i < numberOfSpotLights; i++){
		if(pass >= 1){ Graphics::setBlending(GLBlend::One, GLBlend::One); }

		SpotLight* spotLight = manager->getComponentByIndex<SpotLight>(i);

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
		if(pass >= 1){ Graphics::setBlending(GLBlend::One, GLBlend::One); }

		PointLight* pointLight = manager->getComponentByIndex<PointLight>(i);

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

	for(int i = 0; i < manager->getNumberOfComponents<LineRenderer>(); i++){
		LineRenderer* lineRenderer = manager->getComponentByIndex<LineRenderer>(i);
		Transform* transform = lineRenderer->getComponent<Transform>();

		Material* material = manager->getAsset<Material>(lineRenderer->materialId);
		Line* line = manager->getLine();

		line->start = lineRenderer->start;
		line->end = lineRenderer->end;

		Graphics::apply(line);

		glm::mat4 model = transform->getModelMatrix();

		Graphics::bind(material, model);
		Graphics::bind(line);
		Graphics::draw(line);
		Graphics::unbind(line);
	}

	Graphics::checkError();
}

void RenderSystem::renderScene()
{
	for(int i = 0; i < manager->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = manager->getComponentByIndex<MeshRenderer>(i);
		Transform* transform = meshRenderer->getComponent<Transform>();

		Material* material = manager->getAsset<Material>(meshRenderer->materialId); 
		Mesh* mesh = manager->getAsset<Mesh>(meshRenderer->meshId);

		glm::mat4 model = transform->getModelMatrix();

		Graphics::bind(material, model);
		Graphics::bind(mesh);
		Graphics::draw(mesh);
		Graphics::unbind(mesh);
	}

	Graphics::checkError();
}