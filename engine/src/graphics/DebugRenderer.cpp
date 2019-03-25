#include "../../include/graphics/DebugRenderer.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GLHandle.h"
#include "../../include/graphics/OpenGL.h"

#include "../../include/components/Camera.h"

using namespace PhysicsEngine;

DebugRenderer::DebugRenderer()
{
	lineBuffer = new SlabBuffer(60000);	
}

DebugRenderer::~DebugRenderer()
{
	delete lineBuffer;
}

void DebugRenderer::init(World* world)
{
	std::cout << "Debug renderer init called" << std::endl;

	this->world = world;

	font.load("C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\Demo\\data\\fonts\\arial.ttf");

	window.x = 0.5f;
	window.y = 0.5f;
	window.width = 0.5f;
	window.height = 0.5f;

	graph.x =  0.75f;
	graph.y = 0.15f;
	graph.width = 0.4f;
	graph.height = 0.1f;
	graph.rangeMin = 0.0f;
	graph.rangeMax = 60.0f;
	graph.numberOfSamples = 40;

	window.init();
	graph.init();

	windowTexture = NULL;

	lineMaterial = world->createAsset<Material>();
	lineShader = world->createAsset<Shader>();
	lineShader->vertexShader = Shader::lineVertexShader;
	lineShader->fragmentShader = Shader::lineFragmentShader;
	lineShader->compile();
	lineMaterial->shaderId = lineShader->assetId;

	Graphics::checkError();
}

void DebugRenderer::update(Input input, GraphicsDebug debug, GraphicsQuery query)
{
	Camera* camera;
	if(world->getNumberOfComponents<Camera>() > 0){
		camera = world->getComponentByIndex<Camera>(0);
	}
	else{
		std::cout << "Warning: No camera found" << std::endl;
		return;
	}

	glViewport(camera->x, camera->y, camera->width, camera->height - 40);

	glClearDepth(1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Graphics::renderText(world, camera, &font, "Number of batches draw calls: " + std::to_string(query.numBatchDrawCalls), 25.0f, 500.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Number of draw calls: " + std::to_string(query.numDrawCalls), 25.0f, 400.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Elapsed time: " + std::to_string(query.totalElapsedTime), 25.0f, 300.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));

	// 	lineBuffer->clear();

	// 	std::vector<float> lines(6);
	// 	for(int i = 0; i < world->getNumberOfComponents<LineRenderer>(); i++){
	// 		LineRenderer* lineRenderer = world->getComponentByIndex<LineRenderer>(i);
	// 		Material* material = world->getAsset<Material>(lineRenderer->materialId);

	// 		Transform* transform = lineRenderer->getComponent<Transform>(world);
	// 		glm::mat4 model = transform->getModelMatrix();

	// 		glm::vec3 start = glm::vec3(model * glm::vec4(lineRenderer->start, 1.0f));
	// 		glm::vec3 end = glm::vec3(model * glm::vec4(lineRenderer->end, 1.0f));

	// 		lines[0] = start.x;
	// 		lines[1] = start.y;
	// 		lines[2] = start.z;

	// 		lines[3] = end.x;
	// 		lines[4] = end.y;
	// 		lines[5] = end.z;

	// 		lineBuffer->add(lines, material);
	// 	}

	// 	if(world->debug){
	// 		lines = world->getPhysicsTree()->getLines();

	// 		lineBuffer->add(lines, lineMaterial);
	// 	}

	// 	while(lineBuffer->hasNext()){
	// 		SlabNode* node = lineBuffer->getNext();
		 
	// 	 	Graphics::apply(node);
	// 		Graphics::bind(world, node->material, glm::mat4(1.0f));
	// 		Graphics::bind(node);
	// 		Graphics::draw(node);
	// 	}

		graph.add(1.0f);

		if(getKeyDown(input, KeyCode::NumPad0)){
			windowTexture = &debug.fbo[0].depthBuffer;
		}
		else if(getKeyDown(input, KeyCode::NumPad1)){
			windowTexture = &debug.fbo[1].colorBuffer;
		}
		else if(getKeyDown(input, KeyCode::NumPad2)){
			windowTexture = &debug.fbo[2].colorBuffer;
		}

		if(windowTexture != NULL){
			Graphics::render(world, &window.shader, windowTexture, glm::mat4(1.0f), window.VAO, 6, NULL);
		}

		Graphics::render(world, &graph.shader, glm::mat4(1.0f), graph.VAO, 6*(graph.numberOfSamples - 1), NULL);

		Graphics::checkError();
}