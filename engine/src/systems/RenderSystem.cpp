#include <cstddef>
#include <ctime>

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/Graphics.h"

#include "../../include/components/Transform.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/DirectionalLight.h"
#include "../../include/components/SpotLight.h"
#include "../../include/components/PointLight.h"
#include "../../include/components/Camera.h"

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/World.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Line.h"
#include "../../include/core/Input.h"
#include "../../include/core/Time.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem()
{
	type = 0;
}

RenderSystem::RenderSystem(std::vector<char> data)
{
	type = 0;
}

RenderSystem::~RenderSystem()
{
	delete graph;
	delete debugWindow;

	delete lineBuffer;
}

void* RenderSystem::operator new(size_t size)
{
	return getAllocator<RenderSystem>().allocate();
}

void RenderSystem::operator delete(void*)
{

}

void RenderSystem::init(World* world)
{
	this->world = world;

	for(int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++){
		Graphics::generate(world->getAssetByIndex<Texture2D>(i));
	}

	for(int i = 0; i < world->getNumberOfAssets<Shader>(); i++){
		Shader* shader = world->getAssetByIndex<Shader>(i);

		shader->compile();

		if(!shader->isCompiled()){
			std::cout << "Shader failed to compile " << i << std::endl;
		}

		Graphics::setUniformBlockToBindingPoint(shader, "CameraBlock", 0);
		Graphics::setUniformBlockToBindingPoint(shader, "DirectionalLightBlock", 2);
		Graphics::setUniformBlockToBindingPoint(shader, "SpotLightBlock", 3);
		Graphics::setUniformBlockToBindingPoint(shader, "PointLightBlock", 4);
	}

	Graphics::checkError();

	// for each loaded mesh in cpu, generate VBO's and VAO's on gpu
	for(int i = 0; i < world->getNumberOfAssets<Mesh>(); i++){
		Mesh* mesh = world->getAssetByIndex<Mesh>(i);

		Graphics::generate(mesh);
	}

	Line* line = world->getLine();
	lineBuffer = new SlabBuffer(60000);

	Graphics::generate(line);

	Graphics::generate(&cameraState);
	Graphics::generate(&directionLightState);
	Graphics::generate(&spotLightState);
	Graphics::generate(&pointLightState);

	// debug 
	graph = new PerformanceGraph(0.75f, 0.15f, 0.4f, 0.1f, 0.0f, 60.0f, 40);
	debugWindow = new DebugWindow(0.5f, 0.5f, 0.5f, 0.5f);

	graphMaterial = world->create<Material>();
	windowMaterial = world->create<Material>();
	normalMapMaterial = world->create<Material>();
	depthMapMaterial = world->create<Material>();
	lineMaterial = world->create<Material>();

	graphShader = world->create<Shader>();
	windowShader = world->create<Shader>();
	normalMapShader = world->create<Shader>();
	depthMapShader = world->create<Shader>();
	lineShader = world->create<Shader>();

	graphShader->vertexShader = Shader::graphVertexShader;
	graphShader->fragmentShader = Shader::graphFragmentShader;
	windowShader->vertexShader = Shader::windowVertexShader;
	windowShader->fragmentShader = Shader::windowFragmentShader;
	normalMapShader->vertexShader = Shader::normalMapVertexShader;
	normalMapShader->fragmentShader = Shader::normalMapFragmentShader;
	depthMapShader->vertexShader = Shader::depthMapVertexShader;
	depthMapShader->fragmentShader = Shader::depthMapFragmentShader;
	lineShader->vertexShader = Shader::lineVertexShader;
	lineShader->fragmentShader = Shader::lineFragmentShader;

	graphShader->compile();
	windowShader->compile();
	normalMapShader->compile();
	depthMapShader->compile();
	lineShader->compile();

	graphMaterial->shaderId = graphShader->assetId;
	windowMaterial->shaderId = windowShader->assetId;
	normalMapMaterial->shaderId = normalMapShader->assetId;
	depthMapMaterial->shaderId = depthMapShader->assetId;
	lineMaterial->shaderId = lineShader->assetId;

	Graphics::generate(graph);
	Graphics::generate(debugWindow);

	fbo.colorBuffer = world->create<Texture2D>();
	fbo.colorBuffer->redefine(1000, 1000, TextureFormat::RGB);
	fbo.depthBuffer = world->create<Texture2D>();
	fbo.depthBuffer->redefine(1000, 1000, TextureFormat::Depth);

	debugMaterial = normalMapMaterial;
	debugBuffer = fbo.colorBuffer;

	windowMaterial->textureId = fbo.colorBuffer->assetId;

	Graphics::generate(&fbo);

	Graphics::enableBlend();
	Graphics::enableDepthTest();
	Graphics::enableCubemaps();
	Graphics::enablePoints();

	Graphics::checkError();
}

void RenderSystem::update(Input input)
{
	//Graphics::beginGPUTimer();

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

		renderScene();

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

		renderScene();

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

		renderScene();
			
		pass++;
	}

	lineBuffer->clear();

	std::vector<float> lines(6);
	for(int i = 0; i < world->getNumberOfComponents<LineRenderer>(); i++){
		LineRenderer* lineRenderer = world->getComponentByIndex<LineRenderer>(i);
		Material* material = world->getAsset<Material>(lineRenderer->materialId);

		Transform* transform = lineRenderer->getComponent<Transform>(world);
		glm::mat4 model = transform->getModelMatrix();

		glm::vec3 start = glm::vec3(model * glm::vec4(lineRenderer->start, 1.0f));
		glm::vec3 end = glm::vec3(model * glm::vec4(lineRenderer->end, 1.0f));

		lines[0] = start.x;
		lines[1] = start.y;
		lines[2] = start.z;

		lines[3] = end.x;
		lines[4] = end.y;
		lines[5] = end.z;

		lineBuffer->add(lines, material);
	}

	// int numDrawCalls = 0;
	// double duration;
	// std::clock_t start = std::clock();

	if(world->debug){
		lines = world->getPhysicsTree()->getLines();

		lineBuffer->add(lines, lineMaterial);

		// write the slow many draw call way of drawing lines here just for testing and comparing perf to slab buffer
		// for(int i = 0; i < lines.size() / 6; i++){
		// 		Line* line = world->getLine();

		// 		line->start.x = lines[6*i];
		// 		line->start.y = lines[6*i + 1];
		// 		line->start.z = lines[6*i + 2];
		// 		line->end.x = lines[6*i + 3];
		// 		line->end.y = lines[6*i + 4];
		// 		line->end.z = lines[6*i + 5];

		// 		Graphics::apply(line);
		// 		Graphics::bind(lineMaterial, glm::mat4(1.0f));
		// 		Graphics::bind(line);
		// 		Graphics::draw(line);

		// 		numDrawCalls++;
		// }
	}

	while(lineBuffer->hasNext()){
		SlabNode* node = lineBuffer->getNext();
	 
	 	Graphics::apply(node);
		Graphics::bind(world, node->material, glm::mat4(1.0f));
		Graphics::bind(node);
		Graphics::draw(node);

		//numDrawCalls++;
	}

	//duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	//std::cout << "number of draw calls: " << numDrawCalls << " duration: " << duration << std::endl;

	//int elapsedGPUTime = Graphics::endGPUTimer();

	if(world->debug){
		if(input.getKeyDown(KeyCode::NumPad0)){
			debugMaterial = normalMapMaterial;
			debugBuffer = fbo.colorBuffer;
		}
		else if(input.getKeyDown(KeyCode::NumPad1)){
			debugMaterial = depthMapMaterial;
			debugBuffer = fbo.depthBuffer;
		}

		Graphics::bind(&fbo);
		Graphics::clearColorBuffer(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
		Graphics::clearDepthBuffer(1.0f);
		renderScene(debugMaterial);
		Graphics::unbind(&fbo);

		if(input.getKeyDown(KeyCode::P)){
			debugBuffer->readPixels();
			std::vector<unsigned char> temp = debugBuffer->getRawTextureData();

			World::writeToBMP("test.bmp", temp, 1000, 1000, 3);
		}

		windowMaterial->textureId = debugBuffer->assetId;

		//glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;
		Graphics::bind(world, windowMaterial, glm::mat4(1.0f));
		Graphics::bind(debugWindow);
		Graphics::draw(debugWindow);

		if(Time::frameCount % 10 == 0){
			float deltaTime = Time::deltaTime;
			float gpuDeltaTime = Time::gpuDeltaTime;

			graph->add(deltaTime);

			Graphics::apply(graph);
		}

		//std::cout << "gpu time: " << elapsedGPUTime << std::endl;
		
		Graphics::bind(world, graphMaterial, glm::mat4(1.0f));
		Graphics::bind(graph);
		Graphics::draw(graph);
		Graphics::unbind(graph);
	}

	Graphics::checkError();
}

void RenderSystem::renderScene()
{
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		Transform* transform = meshRenderer->getComponent<Transform>(world);

		Material* material = world->getAsset<Material>(meshRenderer->materialId); 
		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

		glm::mat4 model = transform->getModelMatrix();

		Graphics::bind(world, material, model);
		Graphics::bind(mesh);
		Graphics::draw(mesh);
		Graphics::unbind(mesh);
	}

	Graphics::checkError();
}

void RenderSystem::renderScene(Material* material)
{
	for(int i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++){
		MeshRenderer* meshRenderer = world->getComponentByIndex<MeshRenderer>(i);
		Transform* transform = meshRenderer->getComponent<Transform>(world);

		Mesh* mesh = world->getAsset<Mesh>(meshRenderer->meshId);

		glm::mat4 model = transform->getModelMatrix();

		Graphics::bind(world, material, model);
		Graphics::bind(mesh);
		Graphics::draw(mesh);
	}

	Graphics::checkError();
}