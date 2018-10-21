#include "../../include/core/Material.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Material::Material()
{
	materialId = -1;
	shaderId = -1;
	textureId = -1;

	shininess = 1.0f;
	ambient = glm::vec3(0.25f, 0.25f, 0.25f);
	diffuse = glm::vec3(0.75f, 0.75f, 0.75f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
}

Material::~Material()
{

}

void Material::setManager(Manager* manager)
{
	this->manager = manager;
}

Shader* Material::getShader()
{
	return manager->getShader(shaderId);
}

Texture2D* Material::getMainTexture()
{
	return manager->getTexture2D(textureId);
}

Texture2D* Material::getNormalMap()
{
	return manager->getTexture2D(normalMapId);
}

Texture2D* Material::getSpecularMap()
{
	return manager->getTexture2D(specularMapId);
}