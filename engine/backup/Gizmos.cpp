// #include <iostream>
// #include "Gizmos.h"
// #include "OpenGL.h"

// #include "../core/Log.h"

// #include "../MeshLoader.h"

// using namespace PhysicsEngine;

// Shader Gizmos::gizmoShader;

// VertexArrayObject Gizmos::sphereVAO;
// VertexArrayObject Gizmos::cubeVAO;
// VertexArrayObject Gizmos::meshVAO;
// VertexArrayObject Gizmos::treeVAO;

// Buffer Gizmos::sphereVBO;
// Buffer Gizmos::cubeVBO;
// Buffer Gizmos::meshVBO;
// Buffer Gizmos::treeVBO;

// Mesh Gizmos::sphereMesh;
// Mesh Gizmos::cubeMesh;

// bool Gizmos::isInitialized;

// glm::mat4 Gizmos::projection;
// glm::mat4 Gizmos::view;

// void Gizmos::init()
// {
// 	isInitialized = false;

// 	if (!gizmoShader.compile("../data/shaders/gizmos.vs", "../data/shaders/gizmos.frag")){
// 		Log::Warn("Gizmos: shader failed to compile");
// 		return;
// 	}

// 	if (!MeshLoader::load("../data/meshes/sphere.txt", sphereMesh)){
// 		Log::Warn("Gizmos: Could not load obj/sphere.txt mesh");	
// 	}

// 	if (!MeshLoader::load("../data/meshes/cube.txt", cubeMesh)){
// 		Log::Warn("Gizmos: Could not load obj/cube.txt mesh");
// 	}

// 	sphereVAO.generate();
// 	sphereVAO.bind();
// 	sphereVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 	sphereVBO.bind();
// 	sphereVBO.setData(&(sphereMesh.vertices[0]), sphereMesh.vertices.size()*sizeof(float));
// 	sphereVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
// 	sphereVAO.unbind();

// 	cubeVAO.generate();
// 	cubeVAO.bind();
// 	cubeVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 	cubeVBO.bind();
// 	cubeVBO.setData(&(cubeMesh.vertices[0]), cubeMesh.vertices.size()*sizeof(float));
// 	cubeVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
// 	cubeVAO.unbind();

// 	// meshVAO.generate();
// 	// meshVAO.bind();
// 	// meshVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 	// meshVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
// 	// meshVAO.unbind();

// 	treeVAO.generate();
// 	treeVAO.bind();
// 	treeVAO.setDrawMode(GL_LINES);
// 	treeVBO.generate(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
// 	treeVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
// 	treeVAO.unbind();

// 	OpenGL::checkError();

// 	isInitialized = true;
// }

// void Gizmos::drawSphere(glm::vec3 centre, float radius, Color color)
// {

// }

// void Gizmos::drawCube(glm::vec3 centre, glm::vec3 size, Color color)
// {

// }

// void Gizmos::drawMesh(Mesh* mesh, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, Color color)
// {

// }

// void Gizmos::drawWireSphere(glm::vec3 centre, float radius, Color color)
// {
// 	if (!isInitialized){
// 		Log::Warn("Gizmos: If you are going to use gizmos then you must call Gizmos::init() first");
// 		return;
// 	}

// 	glm::mat4 model = glm::mat4();
// 	model = glm::translate(model, centre);
// 	model = glm::scale(model, glm::vec3(radius, radius, radius));

// 	draw(sphereVAO, (int)sphereMesh.vertices.size() / 3, model, color);
// }

// void Gizmos::drawWireCube(glm::vec3 centre, glm::vec3 size, Color color)
// {
// 	if (!isInitialized){
// 		Log::Warn("Gizmos: If you are going to use gizmos then you must call Gizmos::init() first");
// 		return;
// 	}
	
// 	glm::mat4 model = glm::mat4();
// 	model = glm::translate(model, centre);
// 	model = glm::scale(model, glm::vec3(size.x, size.y, size.z));

// 	draw(cubeVAO, (int)cubeMesh.vertices.size() / 3, model, color);
// }

// void Gizmos::drawWireMesh(Mesh* mesh, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, Color color)
// {
// 	if (!isInitialized){
// 		Log::Warn("Gizmos: If you are going to use gizmos then you must call Gizmos::init() first");
// 		return;
// 	}

// 	glm::mat4 model = glm::mat4();
// 	model = glm::translate(model, position);
// 	model = glm::scale(model, scale);
// 	//TODO: add rotation once I implement quaternions

// 	meshVBO.bind();
// 	meshVBO.setData(&(mesh->vertices[0]), mesh->vertices.size()*sizeof(float));
// 	meshVBO.unbind();

// 	draw(meshVAO, (int)mesh->vertices.size() / 3, model, color);
// }

// void Gizmos::drawOcttree(Octtree* tree, Color color)
// {
// 	if (!isInitialized){
// 		Log::Warn("Gizmos: If you are going to use gizmos then you must call Gizmos::init() first");
// 		return;
// 	}

// 	glm::mat4 model = glm::mat4(1.0f);
// 	model[3][3] = 1.0f;

// 	std::vector<float> lines = tree->getWireframe();

// 	// Log::Info("lines: %d", lines.size());
// 	// for(unsigned int i = 0; i < lines.size() / 3; i++){
// 	// 	Log::Info("[i]: %f  %f  %f", lines[3*i], lines[3*i+1], lines[3*i+2]);
// 	// }

// 	treeVBO.bind();
// 	treeVBO.setData(&lines[0], lines.size()*sizeof(float));
// 	treeVBO.unbind();

// 	draw(treeVAO, (int)lines.size() / 3, model, color);
// }

// void Gizmos::draw(VertexArrayObject vao, int numVertices, glm::mat4 model, Color color)
// {
// 	vao.bind();

// 	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

// 	gizmoShader.bind();

// 	gizmoShader.setMat4("projection", projection);
// 	gizmoShader.setMat4("view", view);
// 	gizmoShader.setMat4("model", model);
// 	gizmoShader.setVec4("color", glm::vec4(color.r, color.g, color.b, color.a));

// 	vao.draw(numVertices);

// 	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

// 	vao.unbind();

// 	OpenGL::checkError();
// }