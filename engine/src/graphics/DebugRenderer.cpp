#include "../../include/graphics/DebugRenderer.h"

using namespace PhysicsEngine;

DebugRenderer::DebugRenderer()
{
	graph = new PerformanceGraph(0.75f, 0.15f, 0.4f, 0.1f, 0.0f, 60.0f, 40);
	debugWindow = new DebugWindow(0.5f, 0.5f, 0.5f, 0.5f);
	lineBuffer = new SlabBuffer(60000);	
}

DebugRenderer::~DebugRenderer()
{
	delete graph;
	delete debugWindow;
	delete lineBuffer;
}

void DebugRenderer::init(World* world)
{
	graphMaterial = world->createAsset<Material>();
	windowMaterial = world->createAsset<Material>();
	normalMapMaterial = world->createAsset<Material>();
	depthMapMaterial = world->createAsset<Material>();
	lineMaterial = world->createAsset<Material>();

	graphShader = world->createAsset<Shader>();
	windowShader = world->createAsset<Shader>();
	normalMapShader = world->createAsset<Shader>();
	depthMapShader = world->createAsset<Shader>();
	lineShader = world->createAsset<Shader>();

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

	fbo.colorBuffer = world->createAsset<Texture2D>();
	fbo.colorBuffer->redefine(1000, 1000, TextureFormat::RGB);
	fbo.depthBuffer = world->createAsset<Texture2D>();
	fbo.depthBuffer->redefine(1000, 1000, TextureFormat::Depth);

	debugMaterial = normalMapMaterial;
	debugBuffer = fbo.colorBuffer;

	windowMaterial->textureId = fbo.colorBuffer->assetId;

	Graphics::generate(&fbo);

	Graphics::checkError();
}

void DebugRenderer::update()
{
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

	// 	if(world->debug){
	// 		if(getKeyDown(input, KeyCode::NumPad0)){
	// 			debugMaterial = normalMapMaterial;
	// 			debugBuffer = fbo.colorBuffer;
	// 		}
	// 		else if(getKeyDown(input, KeyCode::NumPad1)){
	// 			debugMaterial = depthMapMaterial;
	// 			debugBuffer = fbo.depthBuffer;
	// 		}

	// 		Graphics::bind(&fbo);
	// 		Graphics::clearColorBuffer(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	// 		Graphics::clearDepthBuffer(1.0f);
	// 		renderScene(debugMaterial);
	// 		Graphics::unbind(&fbo);

	// 		if(getKeyDown(input, KeyCode::P)){
	// 			debugBuffer->readPixels();
	// 			std::vector<unsigned char> temp = debugBuffer->getRawTextureData();

	// 			World::writeToBMP("test.bmp", temp, 1000, 1000, 3);
	// 		}

	// 		windowMaterial->textureId = debugBuffer->assetId;

	// 		//glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;
	// 		Graphics::bind(world, windowMaterial, glm::mat4(1.0f));
	// 		Graphics::bind(debugWindow);
	// 		Graphics::draw(debugWindow);

	// 		if(Time::frameCount % 10 == 0){
	// 			float deltaTime = Time::deltaTime;
	// 			float gpuDeltaTime = Time::gpuDeltaTime;

	// 			graph->add(deltaTime);

	// 			Graphics::apply(graph);
	// 		}
			
	// 		Graphics::bind(world, graphMaterial, glm::mat4(1.0f));
	// 		Graphics::bind(graph);
	// 		Graphics::draw(graph);
	// 		Graphics::unbind(graph);
	// 	}

	// 	Graphics::checkError();
}