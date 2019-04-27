#include "../../include/graphics/DebugRenderer.h"
#include "../../include/graphics/Graphics.h"

#include "../../include/components/Camera.h"

using namespace PhysicsEngine;

DebugRenderer::DebugRenderer()
{
}

DebugRenderer::~DebugRenderer()
{
}

void DebugRenderer::init(World* world)
{
	std::cout << "Debug renderer init called" << std::endl;

	this->world = world;

	font.load("C:\\Users\\James\\Documents\\PhysicsEngine\\sample_project\\Demo\\Demo\\data\\fonts\\arial.ttf");

	window.x = 0.0f;
	window.y = 0.0f;
	window.width = 1.0f;
	window.height = 1.0f;

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

	// line buffer for all static colliders (useful for debugging)
	std::vector<float> colliderLines;
	for(int i = 0; i < world->getNumberOfComponents<SphereCollider>(); i++){
		SphereCollider* collider = world->getComponentByIndex<SphereCollider>(i);

		std::vector<float> lines = collider->getLines();
		for(size_t j = 0; j < lines.size(); j++){
			colliderLines.push_back(lines[j]);
		}
	}

	for(int i = 0; i < world->getNumberOfComponents<BoxCollider>(); i++){
		BoxCollider* collider = world->getComponentByIndex<BoxCollider>(i);

		std::vector<float> lines = collider->getLines();
		for(size_t j = 0; j < lines.size(); j++){
			colliderLines.push_back(lines[j]);
		}
	}

	colliderBuffer.init(colliderLines);

	std::vector<float> lines = world->getStaticPhysicsGrid()->getLines();

	std::cout << "Number of lines: " << lines.size() << std::endl;

	buffer.init(lines);

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

	std::string title = "Debug";

	if(world->debugView == 0){
		windowTexture = &debug.fbo[0].depthBuffer;
		title = "Depth";
	}
	else if(world->debugView == 1){
		windowTexture = &debug.fbo[1].colorBuffer;
		title = " Normals";
	}
	else if(world->debugView == 2){
		windowTexture = &debug.fbo[2].colorBuffer;
		title = "Overdraw";
	}
	else if(world->debugView == 3){
		windowTexture = &debug.fbo[3].colorBuffer;
		title = "Colliders";
	}

	glBindFramebuffer(GL_FRAMEBUFFER, debug.fbo[world->debugView].handle);

	if(world->debugView == 3){
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_DEPTH_TEST);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
		Graphics::render(world, &colliderBuffer.shader, glm::mat4(1.0f), colliderBuffer.VAO, GL_LINES, (GLsizei)colliderBuffer.size / 3, NULL);
		Graphics::render(world, &buffer.shader, glm::mat4(1.0f), buffer.VAO, GL_LINES, (GLsizei)buffer.size / 3, NULL);
	}

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Graphics::renderText(world, camera, &font, title, 450.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Number of batches draw calls: " + std::to_string(query.numBatchDrawCalls), 25.0f, 500.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Number of draw calls: " + std::to_string(query.numDrawCalls), 25.0f, 400.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Elapsed time: " + std::to_string(query.totalElapsedTime), 25.0f, 300.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if(windowTexture != NULL){
		Graphics::render(world, &window.shader, windowTexture, glm::mat4(1.0f), window.VAO, 6, NULL);
	}

	Graphics::render(world, &graph.shader, glm::mat4(1.0f), graph.VAO, GL_TRIANGLES, 6*(graph.numberOfSamples - 1), NULL);

	Graphics::checkError();
}