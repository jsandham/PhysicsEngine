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
	/*textureId = Guid::INVALID;
	normalMapId = Guid::INVALID;
	specularMapId = Guid::INVALID;

	shininess = 1.0f;
	ambient = glm::vec3(0.25f, 0.25f, 0.25f);
	diffuse = glm::vec3(0.75f, 0.75f, 0.75f);
	specular = glm::vec3(1.0f, 1.0f, 1.0f);
	color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);*/

	shaderChanged = true;
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
	header.uniformCount = uniforms.size();
	/*header.textureId = textureId;
	header.normalMapId = normalMapId;
	header.specularMapId = specularMapId;

	header.shininess = shininess;
	header.ambient = ambient;
	header.diffuse = diffuse;
	header.specular = specular;
	header.color = color;*/

	size_t numberOfBytes = sizeof(MaterialHeader) + uniforms.size() * sizeof(ShaderUniform);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(MaterialHeader));

	size_t startIndex = sizeof(MaterialHeader);
	for (size_t i = 0; i < uniforms.size(); i++) {
		memcpy(&data[startIndex], &uniforms[i], sizeof(ShaderUniform));
		startIndex += sizeof(ShaderUniform);
	}

	return data;
}

void Material::deserialize(std::vector<char> data)
{
	uniforms.clear();

	MaterialHeader* header = reinterpret_cast<MaterialHeader*>(&data[0]);

	assetId = header->assetId;
	shaderId = header->shaderId;
	uniforms.resize(header->uniformCount);

	std::string test = shaderId.toString();
	std::string aaaa = assetId.toString();
	/*textureId = header->textureId;
	normalMapId = header->normalMapId;
	specularMapId = header->specularMapId;

	shininess = header->shininess;
	ambient = header->ambient;
	diffuse = header->diffuse;
	specular = header->specular;
	color = header->color;*/

	//size_t startIndex = sizeof(MaterialHeader);

	//uniforms.clear();
	size_t startIndex = sizeof(MaterialHeader);
	for (size_t i = 0; i < uniforms.size(); i++) {
		ShaderUniform* uniform = reinterpret_cast<ShaderUniform*>(&data[startIndex]);

		uniforms[i] = *uniform;

		std::string test1 = uniforms[i].name;
		if (test1 == "material.mainTexture") {
			std::string test2 = (*reinterpret_cast<Guid*>(uniforms[i].data)).toString();
		}

		startIndex += sizeof(ShaderUniform);
	}

	shaderChanged = true;
}

void Material::apply(World* world)
{
	Shader* shader = world->getAsset<Shader>(shaderId);

	int textureSlot = 0;
	for (size_t i = 0; i < uniforms.size(); i++) {
		if (uniforms[i].type == GL_SAMPLER_2D) {

			Texture2D* texture = world->getAsset<Texture2D>(*reinterpret_cast<Guid*>(uniforms[i].data));
			if(texture != NULL) {
				shader->setInt(uniforms[i].name, textureSlot);

				glActiveTexture(GL_TEXTURE0 + textureSlot);
				glBindTexture(GL_TEXTURE_2D, (GLuint)texture->handle.handle);

				textureSlot++;
			}
		}
		
		/*if (uniforms[i].type == GL_INT ) {
			shader->setInt(uniforms[i].name, *reinterpret_cast<int*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT) {
			shader->setFloat(uniforms[i].name, *reinterpret_cast<float*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_VEC2) {
			shader->setVec2(uniforms[i].name, *reinterpret_cast<glm::vec2*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_VEC3) {
			shader->setVec3(uniforms[i].name, *reinterpret_cast<glm::vec3*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_VEC4) {
			shader->setVec4(uniforms[i].name, *reinterpret_cast<glm::vec4*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_MAT2) {
			shader->setMat2(uniforms[i].name, *reinterpret_cast<glm::mat2*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_MAT3) {
			shader->setMat3(uniforms[i].name, *reinterpret_cast<glm::mat3*>(uniforms[i].data));
		}
		else if (uniforms[i].type == GL_FLOAT_MAT4) {
			shader->setMat4(uniforms[i].name, *reinterpret_cast<glm::mat4*>(uniforms[i].data));
		}*/
	}


	shader->setFloat("material.shininess", 1.0f);
	shader->setVec3("material.ambient", glm::vec3(0.25f, 0.25f, 0.25f));
	shader->setVec3("material.diffuse", glm::vec3(0.75f, 0.75f, 0.75f));
	shader->setVec3("material.specular", glm::vec3(1.0f, 1.0f, 1.0f));
}

void Material::onShaderChanged(World* world)
{
	Shader* shader = world->getAsset<Shader>(shaderId);

	if (shader == NULL) {
		return;
	}

	if (!shader->isCompiled()) {
		return;
	}

	std::string test = shaderId.toString();

	// the uniform data serialized my not be in the same order as the uniforms returned from the 
	// shader (the serialized uniforms are in alphabetical order by name while the uniforms reported 
	// by the shader are in the order in which they are declared in the shader). Therefore need to 
	// correct for this by updating shader reported uniforms with the serialized uniforms 
	std::vector<ShaderUniform> shaderUniforms = shader->getUniforms();
	for (size_t i = 0; i < shaderUniforms.size(); i++) {
		for (size_t j = 0; j < uniforms.size(); j++) {
			if (std::string(shaderUniforms[i].name) == std::string(uniforms[j].name)) {
				for (size_t k = 0; k < 64; k++) {
					shaderUniforms[i].data[k] = uniforms[j].data[k];
				}

				break;
			}
		}
	}

	uniforms = shaderUniforms;

	shaderChanged = false;
}

bool Material::hasShaderChanged() const
{
	return shaderChanged;
}

std::vector<ShaderUniform> Material::getUniforms() const
{
	return uniforms;
}

void Material::setBool(std::string name, bool value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(bool));
	}
}

void Material::setInt(std::string name, int value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(int));
	}
}

void Material::setFloat(std::string name, float value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(float));
	}
}

void Material::setColor(std::string name, const Color& color)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy((void*)uniforms[index].data, &color, sizeof(Color));
	}
}

void Material::setVec2(std::string name, const glm::vec2& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC2) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec2));
	}
}

void Material::setVec3(std::string name, const glm::vec3& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC3) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec3));
	}
}

void Material::setVec4(std::string name, const glm::vec4& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec4));
	}
}

void Material::setMat2(std::string name, const glm::mat2& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT2) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat2));
	}
}

void Material::setMat3(std::string name, const glm::mat3& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT3) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat3));
	}
}

void Material::setMat4(std::string name, const glm::mat4& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT4) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat4));
	}
}

void Material::setTexture(std::string name, const Guid& textureId)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && uniforms[index].type == GL_SAMPLER_2D) {
		memcpy((void*)uniforms[index].data, &textureId, sizeof(Guid));
	}
}

void Material::setBool(int nameLocation, bool value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(bool));
	}
}

void Material::setInt(int nameLocation, int value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(int));
	}
}

void Material::setFloat(int nameLocation, float value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT) {
		memcpy((void*)uniforms[index].data, &value, sizeof(float));
	}
}

void Material::setColor(int nameLocation, const Color& color)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy((void*)uniforms[index].data, &color, sizeof(Color));
	}
}

void Material::setVec2(int nameLocation, const glm::vec2& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC2) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec2));
	}
}

void Material::setVec3(int nameLocation, const glm::vec3& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC3) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec3));
	}
}

void Material::setVec4(int nameLocation, const glm::vec4& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy((void*)uniforms[index].data, &vec, sizeof(glm::vec4));
	}
}

void Material::setMat2(int nameLocation, const glm::mat2& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT2) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat2));
	}
}

void Material::setMat3(int nameLocation, const glm::mat3& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT3) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat3));
	}
}

void Material::setMat4(int nameLocation, const glm::mat4& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT4) {
		memcpy((void*)uniforms[index].data, &mat, sizeof(glm::mat4));
	}
}

void Material::setTexture(int nameLocation, const Guid& textureId)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && uniforms[index].type == GL_SAMPLER_2D) {
		memcpy((void*)uniforms[index].data, &textureId, sizeof(textureId));
	}
}

bool Material::getBool(std::string name) const
{
	int index = findIndexOfUniform(name);
	bool value = false;
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy(&value, uniforms[index].data, sizeof(bool));
	}

	return value;
}

int Material::getInt(std::string name) const
{
	int index = findIndexOfUniform(name);
	int value = false;
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy(&value, uniforms[index].data, sizeof(int));
	}

	return value;
}

float Material::getFloat(std::string name) const
{
	int index = findIndexOfUniform(name);
	float value = false;
	if (index != -1 && uniforms[index].type == GL_FLOAT) {
		memcpy(&value, uniforms[index].data, sizeof(float));
	}

	return value;
}

Color Material::getColor(std::string name) const
{
	int index = findIndexOfUniform(name);
	Color color = Color(0, 0, 0, 255);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy(&color, uniforms[index].data, sizeof(Color));
	}

	return color;
}

glm::vec2 Material::getVec2(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec2 vec = glm::vec2(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC2) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec2));
	}

	return vec;
}

glm::vec3 Material::getVec3(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec3 vec = glm::vec3(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC3) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec3));
	}

	return vec;
}

glm::vec4 Material::getVec4(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec4 vec = glm::vec4(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec4));
	}

	return vec;
}

glm::mat2 Material::getMat2(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat2 mat = glm::mat2(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT2) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat2));
	}

	return mat;
}

glm::mat3 Material::getMat3(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat3 mat = glm::mat3(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT3) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat3));
	}

	return mat;
}

glm::mat4 Material::getMat4(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat4 mat = glm::mat4(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT4) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat4));
	}

	return mat;
}

Guid Material::getTexture(std::string name) const
{
	int index = findIndexOfUniform(name);
	Guid textureId = Guid::INVALID;
	if (index != -1 && uniforms[index].type == GL_SAMPLER_2D) {
		memcpy(&textureId, uniforms[index].data, sizeof(Guid));
	}

	return textureId;
}

bool Material::getBool(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	bool value = false;
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy(&value, uniforms[index].data, sizeof(bool));
	}

	return value;
}

int Material::getInt(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	int value = 0;
	if (index != -1 && uniforms[index].type == GL_INT) {
		memcpy(&value, uniforms[index].data, sizeof(int));
	}

	return value;
}

float Material::getFloat(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	float value = 0.0f;
	if (index != -1 && uniforms[index].type == GL_FLOAT) {
		memcpy(&value, uniforms[index].data, sizeof(float));
	}

	return value;
}

Color Material::getColor(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	Color color = Color(0, 0, 0, 255);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy(&color, uniforms[index].data, sizeof(Color));
	}

	return color;
}

glm::vec2 Material::getVec2(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec2 vec = glm::vec2(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC2) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec2));
	}

	return vec;
}

glm::vec3 Material::getVec3(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec3 vec = glm::vec3(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC3) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec3));
	}

	return vec;
}

glm::vec4 Material::getVec4(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec4 vec = glm::vec4(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_VEC4) {
		memcpy(&vec, uniforms[index].data, sizeof(glm::vec4));
	}

	return vec;
}

glm::mat2 Material::getMat2(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat2 mat = glm::mat2(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT2) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat2));
	}

	return mat;
}

glm::mat3 Material::getMat3(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat3 mat = glm::mat3(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT3) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat3));
	}

	return mat;
}

glm::mat4 Material::getMat4(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat4 mat = glm::mat4(0.0f);
	if (index != -1 && uniforms[index].type == GL_FLOAT_MAT4) {
		memcpy(&mat, uniforms[index].data, sizeof(glm::mat4));
	}

	return mat;
}

Guid Material::getTexture(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	Guid textureId = Guid::INVALID;
	if (index != -1 && uniforms[index].type == GL_SAMPLER_2D) {
		memcpy(&textureId, uniforms[index].data, sizeof(Guid));
	}

	return textureId;
}

int Material::findIndexOfUniform(std::string name) const
{
	for (size_t i = 0; i < uniforms.size(); i++) {
		if (name == uniforms[i].name) {
			return (int)i;
		}
	}

	return -1;
}

int Material::findIndexOfUniform(int nameLocation) const
{
	for (size_t i = 0; i < uniforms.size(); i++) {
		if (nameLocation == uniforms[i].location) {
			return (int)i;
		}
	}

	return -1;
}












void Material::setUniformsEditorOnly(std::vector<ShaderUniform> uniforms)
{
	this->uniforms = uniforms;
}