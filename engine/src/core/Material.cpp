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
	color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

Material::Material(std::vector<char> data)
{
	deserialize(data);
}

Material::~Material()
{

}

std::vector<char> Material::serialize()
{
	MaterialHeader header;
	header.assetId = assetId;
	header.shaderId = shaderId;
	header.textureId = textureId;
	header.normalMapId = normalMapId;
	header.specularMapId = specularMapId;

	header.shininess = shininess;
	header.ambient = ambient;
	header.diffuse = diffuse;
	header.specular = specular;
	header.color = color;

	size_t numberOfBytes = sizeof(MaterialHeader);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(MaterialHeader));

	return data;
}

void Material::deserialize(std::vector<char> data)
{
	MaterialHeader* header = reinterpret_cast<MaterialHeader*>(&data[0]);

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
}