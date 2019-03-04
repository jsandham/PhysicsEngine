#include <iostream>

#include "../../include/core/PoolAllocator.h"
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

Material::Material(std::vector<char> data)
{
	size_t index = sizeof(int);
	MaterialHeader* header = reinterpret_cast<MaterialHeader*>(&data[index]);

	assetId = header->assetId;
	shaderId = header->shaderId;
	textureId = header->textureId;
	normalMapId = header->normalMapId;
	specularMapId = header->specularMapId;

	shininess = header->shininess;
	ambient = header->ambient;
	diffuse = header->diffuse;
	specular = header->specular;
	color = header->color;

	index += sizeof(MaterialHeader);

	std::cout << "material index: " << index << " data size: " << data.size() << std::endl;
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