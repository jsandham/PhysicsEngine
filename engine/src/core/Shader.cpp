#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Shader.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Shader::Shader()
{
	vertexShader = "";
	fragmentShader = "";
	geometryShader = "";

	assetId = Guid::INVALID;

	allCompiled = false;
	activeProgramIndex = -1;
}

Shader::Shader(std::vector<char> data)
{
	deserialize(data);

	allCompiled = false;
	activeProgramIndex = -1;
}

Shader::~Shader()
{

}

std::vector<char> Shader::serialize()
{
	ShaderHeader header;
	header.shaderId = assetId;
	header.vertexShaderSize = vertexShader.length();
	header.geometryShaderSize = geometryShader.length();
	header.fragmentShaderSize = fragmentShader.length();

	size_t numberOfBytes = sizeof(ShaderHeader) + 
						sizeof(char) * vertexShader.length() +
						sizeof(char) * fragmentShader.length() +
						sizeof(char) * geometryShader.length();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(ShaderHeader);
	size_t start3 = start2 + sizeof(char) * vertexShader.length();
	size_t start4 = start3 + sizeof(char) * geometryShader.length();
	size_t start5 = start4 + sizeof(char) * fragmentShader.length();

	memcpy(&data[start1], &header, sizeof(ShaderHeader));
	memcpy(&data[start2], vertexShader.c_str(), sizeof(char) * vertexShader.length());
	memcpy(&data[start3], geometryShader.c_str(), sizeof(char) * geometryShader.length());
	memcpy(&data[start4], fragmentShader.c_str(), sizeof(char) * fragmentShader.length());

	return data;
}

void Shader::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(ShaderHeader);

	ShaderHeader* header = reinterpret_cast<ShaderHeader*>(&data[start1]);

	assetId = header->shaderId;

	size_t vertexShaderSize = header->vertexShaderSize;
	size_t geometryShaderSize = header->geometryShaderSize;
	size_t fragmentShaderSize = header->fragmentShaderSize;

	std::vector<char>::iterator start = data.begin();
	std::vector<char>::iterator end = data.begin();
	start += start2;
	end += start2 + vertexShaderSize;

	vertexShader = std::string(start, end);

	start +=vertexShaderSize;
	end += geometryShaderSize;

	geometryShader = std::string(start, end);

	start += geometryShaderSize;
	end += fragmentShaderSize;

	fragmentShader = std::string(start, end);
}

bool Shader::isCompiled() const
{
	return allCompiled;
}

bool Shader::contains(int variant) const
{
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].variant == variant) {
			return true;
		}
	}

	return false;
}

void Shader::add(int variant)
{
	bool variantFound = false;
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].variant == variant) {
			variantFound = true;
			break;
		}
	}

	if (!variantFound) {
		ShaderProgram program;
		program.version = ShaderVersion::GL430;
		program.compiled = false;
		program.variant = variant;
		program.handle = 0;

		programs.push_back(program);

		allCompiled = false;
	}
}

void Shader::remove(int variant)
{
	int index = -1;
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].variant == variant) {
			index = (int)i;
			break;
		}
	}

	if (index != -1) {
		programs.erase(programs.begin() + index);
	}
}

void Shader::compile()
{
	// compile shader for given variant
	GLint success = 0;
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].compiled) {
			continue;
		}

		std::string version;
		if(programs[i].version == ShaderVersion::GL330) { 
			version = "#version 330 core\n"; 
		}
		else{ 
			version = "#version 430 core\n";
		}

		std::string defines;
		if (programs[i].variant & ShaderVariant::Directional) { defines += "#define DIRECTIONALLIGHT\n"; }
		if (programs[i].variant & ShaderVariant::Spot) { defines += "#define SPOTLIGHT\n"; }
		if (programs[i].variant & ShaderVariant::Point) { defines += "#define POINTLIGHT\n"; }
		if (programs[i].variant & ShaderVariant::HardShadows) { defines += "#define HARDSHADOWS\n"; }
		if (programs[i].variant & ShaderVariant::SoftShadows) { defines += "#define SOFTSHADOWS\n"; }
		if (programs[i].variant & ShaderVariant::SSAO) { defines += "#define SSAO\n"; }
		if (programs[i].variant & ShaderVariant::Cascade) { defines += "#define CASCADE\n"; }

		std::string preProcessedVertexShader = version + defines + vertexShader;
		std::string preProcessedGeometryShader = version + defines + geometryShader;
		std::string preProcessedFragmentShader = version + defines + fragmentShader;

		const GLchar* vertexShaderCharPtr = preProcessedVertexShader.c_str();
		const GLchar* geometryShaderCharPtr = preProcessedGeometryShader.c_str();
		const GLchar* fragmentShaderCharPtr = preProcessedFragmentShader.c_str();

		GLuint vertexShaderObj = 0;
		GLuint fragmentShaderObj = 0;
		GLuint geometryShaderObj = 0;
		GLchar infoLog[512];

		// vertex shader
		vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShaderObj, 1, &vertexShaderCharPtr, NULL);
		glCompileShader(vertexShaderObj);
		glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertexShaderObj, 512, NULL, infoLog);
			std::string message = "Shader: Vertex shader compilation failed\n";
			Log::error(message.c_str());

			return;
		}

		// fragment shader
		fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShaderObj, 1, &fragmentShaderCharPtr, NULL);
		glCompileShader(fragmentShaderObj);
		glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShaderObj, 512, NULL, infoLog);
			std::string message = "Shader: Fragment shader compilation failed\n";
			Log::error(message.c_str());
			return;
		}

		// geometry shader
		if (!geometryShader.empty()) {
			geometryShaderObj = glCreateShader(GL_GEOMETRY_SHADER);
			glShaderSource(geometryShaderObj, 1, &geometryShaderCharPtr, NULL);
			glCompileShader(geometryShaderObj);
			glGetShaderiv(geometryShaderObj, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(geometryShaderObj, 512, NULL, infoLog);
				std::string message = "Shader: Geometry shader compilation failed\n";
				Log::error(message.c_str());
				return;
			}
		}

		// shader program
		programs[i].handle = glCreateProgram();
		glAttachShader(programs[i].handle, vertexShaderObj);
		glAttachShader(programs[i].handle, fragmentShaderObj);
		if (geometryShaderObj != 0) {
			glAttachShader(programs[i].handle, geometryShaderObj);
		}

		glLinkProgram(programs[i].handle);
		glGetProgramiv(programs[i].handle, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(programs[i].handle, 512, NULL, infoLog);
			std::string message = "Shader: Shader program linking failed\n";
			Log::error(message.c_str());
			return;
		}
		glDeleteShader(vertexShaderObj);
		glDeleteShader(fragmentShaderObj);
		if (!geometryShader.empty()) {
			glDeleteShader(geometryShaderObj);
		}

		programs[i].compiled = true;
	}

	allCompiled = true;
}

void Shader::use(int variant)
{
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].variant == variant) {
			activeProgramIndex = (int)i;
			glUseProgram(programs[i].handle);
			return;
		}
	}

	std::string message = "Error: Could not find shader variant " + std::to_string(variant) + "\n";
	Log::error(message.c_str());
}

void Shader::unuse()
{
	activeProgramIndex = -1;
	glUseProgram(0);
}

void Shader::setUniformBlock(std::string blockName, int bindingPoint) const
{
	//set uniform block on all shader program
	for (size_t i = 0; i < programs.size(); i++) {
		GLuint blockIndex = glGetUniformBlockIndex(programs[i].handle, blockName.c_str());
		if (blockIndex != GL_INVALID_INDEX) {
			glUniformBlockBinding(programs[i].handle, blockIndex, bindingPoint);
		}
	}
}

int Shader::findUniformLocation(std::string name) const
{
	if(activeProgramIndex != -1){
		return glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
	}

	return -1;
}

void Shader::setBool(std::string name, bool value) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, (int)value);
		}
	}
}

void Shader::setInt(std::string name, int value) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, value);
		}
	}
}

void Shader::setFloat(std::string name, float value) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform1f(locationIndex, value);
		}
	}
}

void Shader::setVec2(std::string name, const glm::vec2 &vec) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform2fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec3(std::string name, const glm::vec3 &vec) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform3fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec4(std::string name, const glm::vec4 &vec) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniform4fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setMat2(std::string name, const glm::mat2 &mat) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix2fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat3(std::string name, const glm::mat3 &mat) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix3fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat4(std::string name, const glm::mat4 &mat) const
{
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix4fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setBool(int nameLocation, bool value) const
{
	if(activeProgramIndex != -1 && nameLocation != -1){
		glUniform1i(nameLocation, (int)value);
	}
}
void Shader::setInt(int nameLocation, int value) const
{
	if(activeProgramIndex != -1 && nameLocation != -1){
		glUniform1i(nameLocation, value);
	}
}

void Shader::setFloat(int nameLocation, float value) const
{
	if(activeProgramIndex != -1 && nameLocation != -1){
		glUniform1f(nameLocation, value);
	}
}

void Shader::setVec2(int nameLocation, const glm::vec2 &vec) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniform2fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec3(int nameLocation, const glm::vec3 &vec) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniform3fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec4(int nameLocation, const glm::vec4 &vec) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniform4fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setMat2(int nameLocation, const glm::mat2 &mat) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat3(int nameLocation, const glm::mat3 &mat) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat4(int nameLocation, const glm::mat4 &mat) const
{
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

bool Shader::getBool(std::string name) const
{
	int value = 0;
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(programs[activeProgramIndex].handle, locationIndex, &value);
		}
	}

	return (bool)value;
}

int Shader::getInt(std::string name) const
{
	int value = 0;
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(programs[activeProgramIndex].handle, locationIndex, &value);
		}
	}

	return value;
}

float Shader::getFloat(std::string name) const
{
	float value = 0.0f;
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetUniformfv(programs[activeProgramIndex].handle, locationIndex, &value);
		}
	}

	return value;
}

glm::vec2 Shader::getVec2(std::string name) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::vec2), &value[0]);
		}
	}

	return value;
}

glm::vec3 Shader::getVec3(std::string name) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::vec3), &value[0]);
		}
	}

	return value;
}

glm::vec4 Shader::getVec4(std::string name) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::vec4), &value[0]);
		}
	}

	return value;
}

glm::mat2 Shader::getMat2(std::string name) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::mat2), &value[0][0]);
		}
	}

	return value;
}

glm::mat3 Shader::getMat3(std::string name) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::mat3), &value[0][0]);
		}
	}

	return value;
}

glm::mat4 Shader::getMat4(std::string name) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (activeProgramIndex != -1) {
		GLint locationIndex = glGetUniformLocation(programs[activeProgramIndex].handle, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(programs[activeProgramIndex].handle, locationIndex, sizeof(glm::mat4), &value[0][0]);
		}
	}

	return value;
}

bool Shader::getBool(int nameLocation) const
{
	int value = 0;
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetUniformiv(programs[activeProgramIndex].handle, nameLocation, &value);
	}

	return (bool)value;
}

int Shader::getInt(int nameLocation) const
{
	int value = 0;
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetUniformiv(programs[activeProgramIndex].handle, nameLocation, &value);
	}

	return value;
}

float Shader::getFloat(int nameLocation) const
{
	float value = 0.0f;
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetUniformfv(programs[activeProgramIndex].handle, nameLocation, &value);
	}

	return value;
}

glm::vec2 Shader::getVec2(int nameLocation) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::vec2), &value[0]);
	}

	return value;
}

glm::vec3 Shader::getVec3(int nameLocation) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::vec3), &value[0]);
	}

	return value;
}

glm::vec4 Shader::getVec4(int nameLocation) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::vec4), &value[0]);
	}

	return value;
}

glm::mat2 Shader::getMat2(int nameLocation) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::mat2), &value[0][0]);
	}

	return value;
}

glm::mat3 Shader::getMat3(int nameLocation) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::mat3), &value[0][0]);
	}

	return value;
}

glm::mat4 Shader::getMat4(int nameLocation) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (activeProgramIndex != -1 && nameLocation != -1) {
		glGetnUniformfv(programs[activeProgramIndex].handle, nameLocation, sizeof(glm::mat4), &value[0][0]);
	}

	return value;
}