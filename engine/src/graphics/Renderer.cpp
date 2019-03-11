#include "../../include/graphics/Renderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/Camera.h"

#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Line.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

Renderer::Renderer()
{

}

Renderer::~Renderer()
{

}

void Renderer::init(World* world)
{
	this->world = world;

	Graphics::enableBlend();
	Graphics::enableDepthTest();

	for(int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++){
		Graphics::generate(world->getAssetByIndex<Texture2D>(i));
	}

	for(int i = 0; i < world->getNumberOfAssets<Shader>(); i++){
		Shader* shader = world->getAssetByIndex<Shader>(i);

		shader->compile();

		if(!shader->isCompiled()){
			std::cout << "Shader failed to compile " << i << " " << shader->assetId.toString() << std::endl;
		}

		std::string uniformBlocks[] = {"CameraBlock", 
									   "DirectionalLightBlock", 
									   "SpotLightBlock", 
									   "PointLightBlock"};

		for(int i = 0; i < 4; i++){
			Graphics::setUniformBlockToBindingPoint(shader, uniformBlocks[i], i);
		}
	}

	// batch all static meshes by material
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		//Transform* transform = meshRenderer->getComponent<Transform>(world);

		//glm::mat4 model = transform->getModelMatrix();

		Material* material = world->getAsset<Material>(meshRenderer->materialId);
		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

		batchManager.add(material, mesh);

	}

	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
	for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		Graphics::generate(mesh);
	}

	Graphics::generate(&cameraState);
	Graphics::generate(&directionLightState);
	Graphics::generate(&spotLightState);
	Graphics::generate(&pointLightState);

	Graphics::checkError();
}

void Renderer::update()
{
	int numberOfDirectionalLights = world->getNumberOfComponents<DirectionalLight>();
	int numberOfSpotLights = world->getNumberOfComponents<SpotLight>();
	int numberOfPointLights = world->getNumberOfComponents<PointLight>();

	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	Graphics::setViewport(camera->x, camera->y, camera->width, camera->height - 40);
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
		DirectionalLight* directionalLight = world->getComponentByIndex<DirectionalLight>(0);

		Graphics::bind(&directionLightState);
		Graphics::setDirLightDirection(&directionLightState, directionalLight->direction);
		Graphics::setDirLightAmbient(&directionLightState, directionalLight->ambient);
		Graphics::setDirLightDiffuse(&directionLightState, directionalLight->diffuse);
		Graphics::setDirLightSpecular(&directionLightState, directionalLight->specular);
		Graphics::unbind(&directionLightState);

		batchManager.render(world);

		pass++;
	}

	for(int i = 0; i < numberOfSpotLights; i++){
		if(pass >= 1){ Graphics::setBlending(GLBlend::One, GLBlend::One); }

		SpotLight* spotLight = world->getComponentByIndex<SpotLight>(i);

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

		batchManager.render(world);

		pass++;
	}

	for(int i = 0; i < numberOfPointLights; i++){
		if(pass >= 1){ Graphics::setBlending(GLBlend::One, GLBlend::One); }

		PointLight* pointLight = world->getComponentByIndex<PointLight>(i);

		Graphics::bind(&pointLightState);
		Graphics::setPointLightPosition(&pointLightState, pointLight->position);
		Graphics::setPointLightAmbient(&pointLightState, pointLight->ambient);
		Graphics::setPointLightDiffuse(&pointLightState, pointLight->diffuse);
		Graphics::setPointLightSpecular(&pointLightState, pointLight->specular);
		Graphics::setPointLightConstant(&pointLightState, pointLight->constant);
		Graphics::setPointLightLinear(&pointLightState, pointLight->linear);
		Graphics::setPointLightQuadratic(&pointLightState, pointLight->quadratic);
		Graphics::unbind(&pointLightState);

		batchManager.render(world);
			
		pass++;
	}

	Graphics::checkError();
}