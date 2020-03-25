#include <iostream>

#include <GL/glew.h>
#include <gl/gl.h>

#include "../../include/core/Material.h"
#include "../../include/core/World.h"
#include "../../include/core/mat_load.h"

using namespace PhysicsEngine;

Material::Material()
{
	mShaderId = Guid::INVALID;

	mShaderChanged = true;
}

Material::Material(std::vector<char> data)
{
	deserialize(data);
}

Material::~Material()
{

}

std::vector<char> Material::serialize() const
{
	return serialize(mAssetId);
}

std::vector<char> Material::serialize(Guid assetId) const
{
	MaterialHeader header;
	header.mAssetId = assetId;
	header.mShaderId = mShaderId;
	header.mUniformCount = mUniforms.size();

	size_t numberOfBytes = sizeof(MaterialHeader) + mUniforms.size() * sizeof(ShaderUniform);

	std::vector<char> data(numberOfBytes);

	memcpy(&data[0], &header, sizeof(MaterialHeader));

	size_t startIndex = sizeof(MaterialHeader);
	for (size_t i = 0; i < mUniforms.size(); i++) {
		memcpy(&data[startIndex], &mUniforms[i], sizeof(ShaderUniform));
		startIndex += sizeof(ShaderUniform);
	}

	return data;
}

void Material::deserialize(std::vector<char> data)
{
	mUniforms.clear();

	MaterialHeader* header = reinterpret_cast<MaterialHeader*>(&data[0]);

	mAssetId = header->mAssetId;
	mShaderId = header->mShaderId;
	mUniforms.resize(header->mUniformCount);
	
	size_t startIndex = sizeof(MaterialHeader);
	for (size_t i = 0; i < mUniforms.size(); i++) {
		ShaderUniform* uniform = reinterpret_cast<ShaderUniform*>(&data[startIndex]);

		mUniforms[i] = *uniform;

		startIndex += sizeof(ShaderUniform);
	}

	mShaderChanged = true;
}

void Material::load(const std::string& filepath)
{
	material_data mat;

	if (mat_load(filepath, mat))
	{
		mUniforms = mat.mUniforms;
		mShaderId = mat.mShaderId;

		mShaderChanged = true;
	}
	else {
		std::string message = "Error: Could not load material " + filepath + "\n";
		Log::error(message.c_str());
	}
}

void Material::apply(World* world)
{
	Shader* shader = world->getAsset<Shader>(mShaderId);

	int textureSlot = 0;
	for (size_t i = 0; i < mUniforms.size(); i++) {

		if (mUniforms[i].mType == GL_SAMPLER_2D) {

			Texture2D* texture = world->getAsset<Texture2D>(*reinterpret_cast<Guid*>(mUniforms[i].mData));
			if(texture != NULL) {
				shader->setInt(mUniforms[i].mName, textureSlot);

				glActiveTexture(GL_TEXTURE0 + textureSlot);
				glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());

				textureSlot++;
			}
		}
		
		if (mUniforms[i].mType == GL_INT ) {
			shader->setInt(mUniforms[i].mName, *reinterpret_cast<int*>(mUniforms[i].mData));
		}
		else if (mUniforms[i].mType == GL_FLOAT) {
			shader->setFloat(mUniforms[i].mName, *reinterpret_cast<float*>(mUniforms[i].mData));
		}
		else if (mUniforms[i].mType == GL_FLOAT_VEC2) {
			shader->setVec2(mUniforms[i].mName, *reinterpret_cast<glm::vec2*>(mUniforms[i].mData));
		}
		else if (mUniforms[i].mType == GL_FLOAT_VEC3) {
			shader->setVec3(mUniforms[i].mName, *reinterpret_cast<glm::vec3*>(mUniforms[i].mData));
		}
		else if (mUniforms[i].mType == GL_FLOAT_VEC4) {
			shader->setVec4(mUniforms[i].mName, *reinterpret_cast<glm::vec4*>(mUniforms[i].mData));
		}
	}
}

void Material::onShaderChanged(World* world)
{
	Shader* shader = world->getAsset<Shader>(mShaderId);

	if (shader == NULL) {
		return;
	}

	if (!shader->isCompiled()) {
		return;
	}

	// the uniform data serialized may not be in the same order as the uniforms returned from the 
	// shader (the serialized uniforms are in alphabetical order by name while the uniforms reported 
	// by the shader are in the order in which they are declared in the shader). Therefore need to 
	// correct for this by updating shader reported uniforms with the serialized uniforms 
	std::vector<ShaderUniform> shaderUniforms = shader->getUniforms();
	for (size_t i = 0; i < shaderUniforms.size(); i++) {
		for (size_t j = 0; j < mUniforms.size(); j++) {
			if (memcmp(shaderUniforms[i].mName, mUniforms[j].mName, 32) == 0) {
				memcpy(shaderUniforms[i].mData, mUniforms[j].mData, 64);

				break;
			}
		}
	}

	mUniforms = shaderUniforms;

	mShaderChanged = false;
}

bool Material::hasShaderChanged() const
{
	return mShaderChanged;
}

void Material::setShaderId(Guid shaderId)
{
	mShaderId = shaderId;
	mShaderChanged = true;
}

Guid Material::getShaderId() const
{
	return mShaderId;
}

std::vector<ShaderUniform> Material::getUniforms() const
{
	return mUniforms;
}

void Material::setBool(std::string name, bool value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(bool));
	}
}

void Material::setInt(std::string name, int value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(int));
	}
}

void Material::setFloat(std::string name, float value)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(float));
	}
}

void Material::setColor(std::string name, const Color& color)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy((void*)mUniforms[index].mData, &color, sizeof(Color));
	}
}

void Material::setVec2(std::string name, const glm::vec2& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC2) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec2));
	}
}

void Material::setVec3(std::string name, const glm::vec3& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC3) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec3));
	}
}

void Material::setVec4(std::string name, const glm::vec4& vec)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec4));
	}
}

void Material::setMat2(std::string name, const glm::mat2& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT2) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat2));
	}
}

void Material::setMat3(std::string name, const glm::mat3& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT3) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat3));
	}
}

void Material::setMat4(std::string name, const glm::mat4& mat)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT4) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat4));
	}
}

void Material::setTexture(std::string name, const Guid& textureId)
{
	int index = findIndexOfUniform(name);
	if (index != -1 && mUniforms[index].mType == GL_SAMPLER_2D) {
		memcpy((void*)mUniforms[index].mData, &textureId, sizeof(Guid));
	}
}

void Material::setBool(int nameLocation, bool value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(bool));
	}
}

void Material::setInt(int nameLocation, int value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(int));
	}
}

void Material::setFloat(int nameLocation, float value)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT) {
		memcpy((void*)mUniforms[index].mData, &value, sizeof(float));
	}
}

void Material::setColor(int nameLocation, const Color& color)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy((void*)mUniforms[index].mData, &color, sizeof(Color));
	}
}

void Material::setVec2(int nameLocation, const glm::vec2& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC2) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec2));
	}
}

void Material::setVec3(int nameLocation, const glm::vec3& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC3) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec3));
	}
}

void Material::setVec4(int nameLocation, const glm::vec4& vec)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy((void*)mUniforms[index].mData, &vec, sizeof(glm::vec4));
	}
}

void Material::setMat2(int nameLocation, const glm::mat2& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT2) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat2));
	}
}

void Material::setMat3(int nameLocation, const glm::mat3& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT3) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat3));
	}
}

void Material::setMat4(int nameLocation, const glm::mat4& mat)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT4) {
		memcpy((void*)mUniforms[index].mData, &mat, sizeof(glm::mat4));
	}
}

void Material::setTexture(int nameLocation, const Guid& textureId)
{
	int index = findIndexOfUniform(nameLocation);
	if (index != -1 && mUniforms[index].mType == GL_SAMPLER_2D) {
		memcpy((void*)mUniforms[index].mData, &textureId, sizeof(textureId));
	}
}

bool Material::getBool(std::string name) const
{
	int index = findIndexOfUniform(name);
	bool value = false;
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy(&value, mUniforms[index].mData, sizeof(bool));
	}

	return value;
}

int Material::getInt(std::string name) const
{
	int index = findIndexOfUniform(name);
	int value = false;
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy(&value, mUniforms[index].mData, sizeof(int));
	}

	return value;
}

float Material::getFloat(std::string name) const
{
	int index = findIndexOfUniform(name);
	float value = false;
	if (index != -1 && mUniforms[index].mType == GL_FLOAT) {
		memcpy(&value, mUniforms[index].mData, sizeof(float));
	}

	return value;
}

Color Material::getColor(std::string name) const
{
	int index = findIndexOfUniform(name);
	Color color = Color(0, 0, 0, 255);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy(&color, mUniforms[index].mData, sizeof(Color));
	}

	return color;
}

glm::vec2 Material::getVec2(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec2 vec = glm::vec2(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC2) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec2));
	}

	return vec;
}

glm::vec3 Material::getVec3(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec3 vec = glm::vec3(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC3) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec3));
	}

	return vec;
}

glm::vec4 Material::getVec4(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::vec4 vec = glm::vec4(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec4));
	}

	return vec;
}

glm::mat2 Material::getMat2(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat2 mat = glm::mat2(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT2) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat2));
	}

	return mat;
}

glm::mat3 Material::getMat3(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat3 mat = glm::mat3(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT3) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat3));
	}

	return mat;
}

glm::mat4 Material::getMat4(std::string name) const
{
	int index = findIndexOfUniform(name);
	glm::mat4 mat = glm::mat4(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT4) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat4));
	}

	return mat;
}

Guid Material::getTexture(std::string name) const
{
	int index = findIndexOfUniform(name);
	Guid textureId = Guid::INVALID;
	if (index != -1 && mUniforms[index].mType == GL_SAMPLER_2D) {
		memcpy(&textureId, mUniforms[index].mData, sizeof(Guid));
	}

	return textureId;
}

bool Material::getBool(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	bool value = false;
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy(&value, mUniforms[index].mData, sizeof(bool));
	}

	return value;
}

int Material::getInt(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	int value = 0;
	if (index != -1 && mUniforms[index].mType == GL_INT) {
		memcpy(&value, mUniforms[index].mData, sizeof(int));
	}

	return value;
}

float Material::getFloat(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	float value = 0.0f;
	if (index != -1 && mUniforms[index].mType == GL_FLOAT) {
		memcpy(&value, mUniforms[index].mData, sizeof(float));
	}

	return value;
}

Color Material::getColor(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	Color color = Color(0, 0, 0, 255);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy(&color, mUniforms[index].mData, sizeof(Color));
	}

	return color;
}

glm::vec2 Material::getVec2(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec2 vec = glm::vec2(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC2) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec2));
	}

	return vec;
}

glm::vec3 Material::getVec3(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec3 vec = glm::vec3(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC3) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec3));
	}

	return vec;
}

glm::vec4 Material::getVec4(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::vec4 vec = glm::vec4(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_VEC4) {
		memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec4));
	}

	return vec;
}

glm::mat2 Material::getMat2(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat2 mat = glm::mat2(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT2) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat2));
	}

	return mat;
}

glm::mat3 Material::getMat3(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat3 mat = glm::mat3(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT3) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat3));
	}

	return mat;
}

glm::mat4 Material::getMat4(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	glm::mat4 mat = glm::mat4(0.0f);
	if (index != -1 && mUniforms[index].mType == GL_FLOAT_MAT4) {
		memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat4));
	}

	return mat;
}

Guid Material::getTexture(int nameLocation) const
{
	int index = findIndexOfUniform(nameLocation);
	Guid textureId = Guid::INVALID;
	if (index != -1 && mUniforms[index].mType == GL_SAMPLER_2D) {
		memcpy(&textureId, mUniforms[index].mData, sizeof(Guid));
	}

	return textureId;
}

int Material::findIndexOfUniform(std::string name) const
{
	for (size_t i = 0; i < mUniforms.size(); i++) {
		if (name == mUniforms[i].mName) {
			return (int)i;
		}
	}

	return -1;
}

int Material::findIndexOfUniform(int nameLocation) const
{
	for (size_t i = 0; i < mUniforms.size(); i++) {
		if (nameLocation == mUniforms[i].mLocation) {
			return (int)i;
		}
	}

	return -1;
}