#include "../../include/core/Material.h"
#include "../../include/core/Manager.h"

using namespace PhysicsEngine;

Material::Material()
{
	shaderId = Guid::INVALID;
	textureId = Guid::INVALID;
	normalMapId = Guid::INVALID;
	specularMapId = Guid::INVALID;

	shininess = 1.0f;
	ambient = glm::vec3(0.25f, 0.25f, 0.25f);
	diffuse = glm::vec3(0.75f, 0.75f, 0.75f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
	color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
}

Material::~Material()
{

}

void Material::load(MaterialData data)
{
	assetId = data.assetId;

	shaderId = data.shaderId;
	textureId = data.textureId;
	normalMapId = data.normalMapId;
	specularMapId = data.specularMapId;

	shininess = data.shininess;
	ambient = data.ambient;
	diffuse = data.diffuse;
	specular = data.specular;
	color = data.color;
}

Shader* Material::getShader()
{
	return manager->getAsset<Shader>(shaderId);
}

Texture2D* Material::getMainTexture()
{
	return manager->getAsset<Texture2D>(textureId);
}

Texture2D* Material::getNormalMap()
{
	return manager->getAsset<Texture2D>(normalMapId);
}

Texture2D* Material::getSpecularMap()
{
	return manager->getAsset<Texture2D>(specularMapId);
}