#include "Skybox.h"

using namespace PhysicsEngine;

std::vector<float> Skybox::triangles = { -10.0f, 10.0f, -10.0f,
										-10.0f, -10.0f, -10.0f,
										10.0f, -10.0f, -10.0f,
										10.0f, -10.0f, -10.0f,
										10.0f, 10.0f, -10.0f,
										-10.0f, 10.0f, -10.0f,

										-10.0f, -10.0f, 10.0f,
										-10.0f, -10.0f, -10.0f,
										-10.0f, 10.0f, -10.0f,
										-10.0f, 10.0f, -10.0f,
										-10.0f, 10.0f, 10.0f,
										-10.0f, -10.0f, 10.0f,

										10.0f, -10.0f, -10.0f,
										10.0f, -10.0f, 10.0f,
										10.0f, 10.0f, 10.0f,
										10.0f, 10.0f, 10.0f,
										10.0f, 10.0f, -10.0f,
										10.0f, -10.0f, -10.0f,

										-10.0f, -10.0f, 10.0f,
										-10.0f, 10.0f, 10.0f,
										10.0f, 10.0f, 10.0f,
										10.0f, 10.0f, 10.0f,
										10.0f, -10.0f, 10.0f,
										-10.0f, -10.0f, 10.0f,

										-10.0f, 10.0f, -10.0f,
										10.0f, 10.0f, -10.0f,
										10.0f, 10.0f, 10.0f,
										10.0f, 10.0f, 10.0f,
										-10.0f, 10.0f, 10.0f,
										-10.0f, 10.0f, -10.0f,

										-10.0f, -10.0f, -10.0f,
										-10.0f, -10.0f, 10.0f,
										10.0f, -10.0f, -10.0f,
										10.0f, -10.0f, -10.0f,
										-10.0f, -10.0f, 10.0f,
										10.0f, -10.0f, 10.0f};

Skybox::Skybox()
{
	skyboxVAO.bind();
	skyboxVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	skyboxVBO.bind();
	/*skyboxVBO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);*/
	skyboxVBO.setData(&triangles, triangles.size()*sizeof(float));

	skyboxVAO.unbind();
}

Skybox::Skybox(Entity *entity)
{
	this->entity = entity;

	skyboxVAO.bind();
	skyboxVAO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
	skyboxVBO.bind();
	/*skyboxVBO.setLayout(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);*/
	skyboxVBO.setData(&triangles, triangles.size()*sizeof(float));

	skyboxVAO.unbind();
}

Skybox::~Skybox()
{

}

Material* Skybox::getMaterial()
{
	return material;
}

void Skybox::setMaterial(Material *material)
{
	this->material = material;
}

void Skybox::draw()
{
	skyboxVAO.bind();
	skyboxVAO.draw((int)triangles.size());
	skyboxVAO.unbind();
}