#include "../../include/core/Material.h"
#include "../../include/core/World.h"

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

Material::Material(unsigned char* data)
{
	
}

Material::~Material()
{

}

void Material::load(MaterialHeader data)
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
	return world->getAsset<Shader>(shaderId);
}

Texture2D* Material::getMainTexture()
{
	return world->getAsset<Texture2D>(textureId);
}

Texture2D* Material::getNormalMap()
{
	return world->getAsset<Texture2D>(normalMapId);
}

Texture2D* Material::getSpecularMap()
{
	return world->getAsset<Texture2D>(specularMapId);
}