#include "../../include/graphics/DebugRenderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Camera.h"

using namespace PhysicsEngine;

DebugRenderer::DebugRenderer()
{
	//lineBuffer = new SlabBuffer(600000);	
}

DebugRenderer::~DebugRenderer()
{
	//delete lineBuffer;
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

	//std::vector<float> lines = world->getStaticPhysicsGrid()->getLines();
	std::vector<float> occupiedLines = world->getStaticPhysicsGrid()->getOccupiedLines();

	std::cout << "Number of lines: " << occupiedLines.size() << std::endl;

	// buffer.init(lines);
	buffer.init(occupiedLines);

	//lineMaterial = world->createAsset<Material>();
	//lineShader = world->createAsset<Shader>();
	//lineShader->vertexShader = Shader::lineVertexShader;
	//lineShader->fragmentShader = Shader::lineFragmentShader;
	//lineShader->compile();
	//lineMaterial->shaderId = lineShader->assetId;

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

	Graphics::render(world, &buffer.shader, glm::mat4(1.0f), buffer.VAO, GL_LINES, (GLsizei)buffer.size / 3, NULL);

	// lineBuffer->clear();

	// std::vector<float> lines = world->getStaticPhysicsGrid()->getLines();

	// lineBuffer->add(lines, lineShader);

	// while(lineBuffer->hasNext()){
	// 	SlabNode* node = lineBuffer->getNext();

	// 	glBindBuffer(GL_ARRAY_BUFFER, node->vbo.handle); 
	// 	glBufferSubData(GL_ARRAY_BUFFER, 0, node->count*sizeof(float), &(node->buffer[0]));

	// 	Graphics::render(world, lineShader, glm::mat4(1.0f), node->vao.handle, GL_LINES, (GLsizei)node->count / 3, NULL);
	// }

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

	Graphics::render(world, &graph.shader, glm::mat4(1.0f), graph.VAO, GL_TRIANGLES, 6*(graph.numberOfSamples - 1), NULL);

	Graphics::checkError();
}