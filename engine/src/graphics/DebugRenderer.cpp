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

void DebugRenderer::init(World* world, bool renderToScreen)
{
	std::cout << "Debug renderer init called" << std::endl;

	this->world = world;
	this->renderToScreen;

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

	colliderBuffer.size = colliderLines.size();
	if(colliderBuffer.size > 0){
		colliderBuffer.shader.vertexShader = Shader::lineVertexShader;
		colliderBuffer.shader.fragmentShader = Shader::lineFragmentShader;

		colliderBuffer.shader.compile();
		std::cout << "collider lines size: " << colliderLines.size() << std::endl;

		glBindVertexArray(colliderBuffer.VAO);
		glBindBuffer(GL_ARRAY_BUFFER, colliderBuffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, colliderLines.size() * sizeof(float), &colliderLines[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
		glBindVertexArray(0);
	}

	std::vector<float> lines = world->getStaticPhysicsGrid()->getLines();

	buffer.size = lines.size();
	if(buffer.size > 0){
		buffer.shader.vertexShader = Shader::lineVertexShader;
		buffer.shader.fragmentShader = Shader::lineFragmentShader;

		buffer.shader.compile();
		std::cout << "lines size: " << lines.size() << std::endl;

		glBindVertexArray(buffer.VAO);
		glBindBuffer(GL_ARRAY_BUFFER, buffer.VBO);
		glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), &lines[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
		glBindVertexArray(0);
	}

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
		if(colliderBuffer.size > 0){
			Graphics::render(world, &colliderBuffer.shader, ShaderVariant::None, glm::mat4(1.0f), colliderBuffer.VAO, GL_LINES, (GLsizei)colliderBuffer.size / 3, NULL);
		}
		if(buffer.size > 0){
			Graphics::render(world, &buffer.shader, ShaderVariant::None, glm::mat4(1.0f), buffer.VAO, GL_LINES, (GLsizei)buffer.size / 3, NULL);
		}
	}

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Graphics::renderText(world, camera, &font, title, 450.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Number of batches draw calls: " + std::to_string(query.numBatchDrawCalls), 25.0f, 980.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Number of draw calls: " + std::to_string(query.numDrawCalls), 25.0f, 955.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Elapsed time: " + std::to_string(query.totalElapsedTime) + " (ms)", 25.0f, 930.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Vertices: " + std::to_string(query.verts), 25.0f, 905.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Triangles: " + std::to_string(query.tris), 25.0f, 880.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Lines: " + std::to_string(query.lines), 25.0f, 855.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));
	Graphics::renderText(world, camera, &font, "Points: " + std::to_string(query.points), 25.0f, 830.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if(windowTexture != NULL){
		Graphics::render(world, &window.shader, ShaderVariant::None, windowTexture, glm::mat4(1.0f), window.VAO, 6, NULL);
	}

	Graphics::render(world, &graph.shader, ShaderVariant::None, glm::mat4(1.0f), graph.VAO, GL_TRIANGLES, 6*(graph.numberOfSamples - 1), NULL);

	Graphics::checkError();
}