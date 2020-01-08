#include <iostream>

#include <GL/glew.h>
#include <gl/gl.h>

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

void Material::use(Shader* shader, RenderObject renderObject)
{
	shader->setFloat("material.shininess", shininess);
	shader->setVec3("material.ambient", ambient);
	shader->setVec3("material.diffuse", diffuse);
	shader->setVec3("material.specular", specular);

	if (renderObject.mainTexture != -1) {
		shader->setInt("material.mainTexture", 0);

		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.mainTexture);
	}

	if (renderObject.normalMap != -1) {
		shader->setInt("material.normalMap", 1);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.normalMap);
	}

	if (renderObject.specularMap != -1) {
		shader->setInt("material.specularMap", 2);

		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.specularMap);
	}
}