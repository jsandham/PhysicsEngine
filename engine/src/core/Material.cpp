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

void Material::onShaderChanged(World* world)
{
	shader = world->getAsset<Shader>(shaderId);
}











void Material::setBool(std::string name, bool value) const
{
	if(shader != NULL){
		shader->setBool(name, value);
	}
}

void Material::setInt(std::string name, int value) const
{
	if(shader != NULL){
		shader->setInt(name, value);
	}
}

void Material::setFloat(std::string name, float value) const
{
	if(shader != NULL){
		shader->setFloat(name, value);
	}
}

void Material::setVec2(std::string name, const glm::vec2& vec) const
{
	if(shader != NULL){
		shader->setVec2(name, vec);
	}
}

void Material::setVec3(std::string name, const glm::vec3& vec) const
{
	if(shader != NULL){
		shader->setVec3(name, vec);
	}
}

void Material::setVec4(std::string name, const glm::vec4& vec) const
{
	if(shader != NULL){
		shader->setVec4(name, vec);
	}
}

void Material::setMat2(std::string name, const glm::mat2& mat) const
{
	if(shader != NULL){
		shader->setMat2(name, mat);
	}
}

void Material::setMat3(std::string name, const glm::mat3& mat) const
{
	if(shader != NULL){
		shader->setMat3(name, mat);
	}
}

void Material::setMat4(std::string name, const glm::mat4& mat) const
{
	if(shader != NULL){
		shader->setMat4(name, mat);
	}
}

void Material::setBool(int nameLocation, bool value) const
{
	if(shader != NULL){
		shader->setBool(nameLocation, value);
	}
}

void Material::setInt(int nameLocation, int value) const
{
	if(shader != NULL){
		shader->setInt(nameLocation, value);
	}
}

void Material::setFloat(int nameLocation, float value) const
{
	if(shader != NULL){
		shader->setFloat(nameLocation, value);
	}
}

void Material::setVec2(int nameLocation, const glm::vec2& vec) const
{
	if(shader != NULL){
		shader->setVec2(nameLocation, vec);
	}
}

void Material::setVec3(int nameLocation, const glm::vec3& vec) const
{
	if(shader != NULL){
		shader->setVec3(nameLocation, vec);
	}
}

void Material::setVec4(int nameLocation, const glm::vec4& vec) const
{
	if(shader != NULL){
		shader->setVec4(nameLocation, vec);
	}
}

void Material::setMat2(int nameLocation, const glm::mat2& mat) const
{
	if(shader != NULL){
		shader->setMat2(nameLocation, mat);
	}
}

void Material::setMat3(int nameLocation, const glm::mat3& mat) const
{
	if(shader != NULL){
		shader->setMat3(nameLocation, mat);
	}
}

void Material::setMat4(int nameLocation, const glm::mat4& mat) const
{
	if(shader != NULL){
		shader->setMat4(nameLocation, mat);
	}
}

bool Material::getBool(std::string name) const
{
	if(shader != NULL){
		return shader->getBool(name);
	}

	return false;
}

int Material::getInt(std::string name) const
{
	if(shader != NULL){
		return shader->getInt(name);
	}

	return 0;
}

float Material::getFloat(std::string name) const
{
	if(shader != NULL){
		return shader->getFloat(name);
	}

	return 0.0f;
}

glm::vec2 Material::getVec2(std::string name) const
{
	if(shader != NULL){
		return shader->getVec2(name);
	}

	return glm::vec2(0.0f);
}

glm::vec3 Material::getVec3(std::string name) const
{
	if(shader != NULL){
		return shader->getVec3(name);
	}

	return glm::vec3(0.0f);
}

glm::vec4 Material::getVec4(std::string name) const
{
	if(shader != NULL){
		return shader->getVec4(name);
	}

	return glm::vec4(0.0f);
}

glm::mat2 Material::getMat2(std::string name) const
{
	if(shader != NULL){
		return shader->getMat2(name);
	}

	return glm::mat2(0.0f);
}

glm::mat3 Material::getMat3(std::string name) const
{
	if(shader != NULL){
		return shader->getMat3(name);
	}

	return glm::mat3(0.0f);
}

glm::mat4 Material::getMat4(std::string name) const
{
	if(shader != NULL){
		return shader->getMat4(name);
	}

	return glm::mat4(0.0f);
}

bool Material::getBool(int nameLocation) const
{
	if(shader != NULL){
		return shader->getBool(nameLocation);
	}

	return false;
}

int Material::getInt(int nameLocation) const
{
	if(shader != NULL){
		return shader->getInt(nameLocation);
	}

	return 0;
}

float Material::getFloat(int nameLocation) const
{
	if(shader != NULL){
		return shader->getFloat(nameLocation);
	}

	return 0.0f;
}

glm::vec2 Material::getVec2(int nameLocation) const
{
	if(shader != NULL){
		return shader->getVec2(nameLocation);
	}

	return glm::vec2(0.0f);
}

glm::vec3 Material::getVec3(int nameLocation) const
{
	if(shader != NULL){
		return shader->getVec3(nameLocation);
	}

	return glm::vec3(0.0f);
}

glm::vec4 Material::getVec4(int nameLocation) const
{
	if(shader != NULL){
		return shader->getVec4(nameLocation);
	}

	return glm::vec4(0.0f);
}

glm::mat2 Material::getMat2(int nameLocation) const
{
	if(shader != NULL){
		return shader->getMat2(nameLocation);
	}

	return glm::mat2(0.0f);
}

glm::mat3 Material::getMat3(int nameLocation) const
{
	if(shader != NULL){
		return shader->getMat3(nameLocation);
	}

	return glm::mat3(0.0f);
}

glm::mat4 Material::getMat4(int nameLocation) const
{
	if(shader != NULL){
		return shader->getMat4(nameLocation);
	}

	return glm::mat4(0.0f);
}