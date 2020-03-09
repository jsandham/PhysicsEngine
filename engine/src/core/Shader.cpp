#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <stack>

#include "../../include/core/Shader.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Shader::Shader()
{
	vertexShader = "";
	fragmentShader = "";
	geometryShader = "";

	assetId = Guid::INVALID;

	allProgramsCompiled = false;
	activeProgram = -1;
}

Shader::Shader(std::vector<char> data)
{
	deserialize(data);

	allProgramsCompiled = false;
	activeProgram = -1;
}

Shader::~Shader()
{

}

std::vector<char> Shader::serialize() const
{
	return serialize(assetId);
}

std::vector<char> Shader::serialize(Guid assetId) const
{
	ShaderHeader header;
	header.shaderId = assetId;
	header.vertexShaderSize = vertexShader.length();
	header.geometryShaderSize = geometryShader.length();
	header.fragmentShaderSize = fragmentShader.length();
	header.numberOfShaderUniforms = uniforms.size();

	size_t numberOfBytes = sizeof(ShaderHeader) +
		sizeof(char) * vertexShader.length() +
		sizeof(char) * fragmentShader.length() +
		sizeof(char) * geometryShader.length() +
		sizeof(ShaderUniform) * uniforms.size();

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
	size_t numberOfShaderUniforms = header->numberOfShaderUniforms;

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

void Shader::load(const std::string& filepath)
{
	std::ifstream in(filepath.c_str());
	std::ostringstream contents; contents << in.rdbuf(); in.close();

	std::string shaderContent = contents.str();

	std::string vertexTag = "VERTEX:";
	std::string geometryTag = "GEOMETRY:";
	std::string fragmentTag = "FRAGMENT:";

	size_t startOfVertexTag = shaderContent.find(vertexTag, 0);
	size_t startOfGeometryTag = shaderContent.find(geometryTag, 0);
	size_t startOfFragmentTag = shaderContent.find(fragmentTag, 0);

	if (startOfVertexTag == std::string::npos || startOfFragmentTag == std::string::npos) {
		std::string message = "Error: Shader must contain both a vertex shader and a fragment shader\n";
		Log::error(message.c_str());
		return;
	}

	std::string vertexShader, geometryShader, fragmentShader;

	if (startOfGeometryTag == std::string::npos) {
		vertexShader = shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfFragmentTag - vertexTag.length());
		geometryShader = "";
		fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
	}
	else {
		vertexShader = shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfGeometryTag - vertexTag.length());
		geometryShader = shaderContent.substr(startOfGeometryTag + geometryTag.length(), startOfFragmentTag - geometryTag.length());
		fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
	}

	// trim left
	size_t firstNotOfIndex;
	firstNotOfIndex = vertexShader.find_first_not_of("\n");
	if (firstNotOfIndex != std::string::npos) {
		vertexShader = vertexShader.substr(firstNotOfIndex);
	}

	firstNotOfIndex = geometryShader.find_first_not_of("\n");
	if (firstNotOfIndex != std::string::npos) {
		geometryShader = geometryShader.substr(firstNotOfIndex);
	}

	firstNotOfIndex = fragmentShader.find_first_not_of("\n");
	if (firstNotOfIndex != std::string::npos) {
		fragmentShader = fragmentShader.substr(firstNotOfIndex);
	}

	// trim right
	size_t lastNotOfIndex;
	lastNotOfIndex = vertexShader.find_last_not_of("\n");
	if (lastNotOfIndex != std::string::npos) {
		vertexShader.erase(lastNotOfIndex + 1);
	}

	lastNotOfIndex = geometryShader.find_last_not_of("\n");
	if (lastNotOfIndex != std::string::npos) {
		geometryShader.erase(lastNotOfIndex + 1);
	}

	lastNotOfIndex = fragmentShader.find_last_not_of("\n");
	if (lastNotOfIndex != std::string::npos) {
		fragmentShader.erase(lastNotOfIndex + 1);
	}

	setVertexShader(vertexShader);
	setGeometryShader(geometryShader);
	setFragmentShader(fragmentShader);
}

void Shader::load(const std::string vertexShader, const std::string fragmentShader, const std::string geometryShader)
{
	setVertexShader(vertexShader);
	setGeometryShader(geometryShader);
	setFragmentShader(fragmentShader);
}

bool Shader::isCompiled() const
{
	return allProgramsCompiled;
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

		allProgramsCompiled = false;
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
	// ensure that all shader programs have the default 'None' program variant
	if (!contains(ShaderVariant::None)) {
		add(static_cast<int>(ShaderVariant::None));
	}

	// determine which variants are possible based on keywords found in shader
	const std::vector<std::string> keywords{ "DIRECTIONALLIGHT", 
											"SPOTLIGHT", 
											"POINTLIGHT", 
											"HARDSHADOWS", 
											"SOFTSHADOWS", 
											"SSAO", 
											"CASCADE" };

	const std::map<const std::string, ShaderVariant> keywordToVariantMap{
		{"DIRECTIONALLIGHT", ShaderVariant::Directional},
		{"SPOTLIGHT", ShaderVariant::Spot},
		{"POINTLIGHT", ShaderVariant::Point},
		{"HARDSHADOWS", ShaderVariant::HardShadows},
		{"SOFTSHADOWS", ShaderVariant::SoftShadows},
		{"SSAO", ShaderVariant::SSAO},
		{"CASCADE", ShaderVariant::Cascade}
	};

	std::vector<ShaderVariant> temp;
	for (size_t i = 0; i < keywords.size(); i++) {
		if (vertexShader.find(keywords[i]) != std::string::npos || 
			geometryShader.find(keywords[i]) != std::string::npos || 
			fragmentShader.find(keywords[i]) != std::string::npos) {
			
			std::map<std::string, ShaderVariant>::const_iterator it = keywordToVariantMap.find(keywords[i]);
			if (it != keywordToVariantMap.end()) {
				temp.push_back(it->second);
			}
		}
	}

	std::set<int> variantsToAdd;
	std::stack<int> stack;
	for (size_t i = 0; i < temp.size(); i++) {
		stack.push(temp[i]);
	}

	while (!stack.empty()) {
		int current = stack.top();
		stack.pop();

		std::set<int>::iterator it = variantsToAdd.find(current);
		if (it == variantsToAdd.end()) {
			variantsToAdd.insert(current);
		}

		for (size_t i = 0; i < temp.size(); i++) {
			if (!(temp[i] & current)) {
				stack.push(current | temp[i]);
			}
		}
	}

	// add variants from keywords found in shader strings in addition to any variants manually added using 'add' method 
	for (std::set<int>::iterator it = variantsToAdd.begin(); it != variantsToAdd.end(); it++) {
		add(*it);
	}

	// compile all shader variants
	GLint success = 0;
	for (size_t i = 0; i < programs.size(); i++) {
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

	allProgramsCompiled = true;

	// find all uniforms and attributes in shader across all variants
	std::set<std::string> uniformNames;
	for (size_t i = 0; i < uniforms.size(); i++) {
		uniformNames.insert(std::string(uniforms[i].name));
	}
	std::set<std::string> attributeNames;
	for (size_t i = 0; i < attributes.size(); i++) {
		attributeNames.insert(std::string(attributes[i].name));
	}

	const GLsizei bufSize = 32; // maximum name length

	// run through all variants and find all uniforms/attributes (and add to sets of known uniforms/attributes if new)
	for (size_t i = 0; i < programs.size(); i++) {
		GLuint program = programs[i].handle;

		GLint uniformCount;
		glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniformCount);
		for (int j = 0; j < uniformCount; j++)
		{
			GLsizei nameLength;
			GLint size;
			GLenum type;
			GLchar name[32];

			glGetActiveUniform(program, (GLuint)j, bufSize, &nameLength, &size, &type, &name[0]);

			ShaderUniform uniform;
			uniform.nameLength = (size_t)nameLength;
			uniform.size = (size_t)size;

			memset(uniform.name, '\0', 32);
			memset(uniform.shortName, '\0', 32);
			memset(uniform.blockName, '\0', 32);

			int indexOfBlockChar = -1;
			for (int k = 0; k < nameLength; k++) {
				uniform.name[k] = name[k];
				if (name[k] == '.') {
					indexOfBlockChar = k;
				}
			}

			uniform.shortName[0] = '\0';
			for (int k = indexOfBlockChar + 1; k < nameLength; k++) {
				uniform.shortName[k - indexOfBlockChar - 1] = name[k];
			}

			uniform.blockName[0] = '\0';
			for (int k = 0; k < indexOfBlockChar; k++) {
				uniform.blockName[k] = name[k];
			}

			uniform.type = type;
			uniform.variant = programs[i].variant;
			uniform.location = findUniformLocation(std::string(uniform.name), program);

			// only add uniform if it wasnt already in array
			std::set<std::string>::iterator it = uniformNames.find(std::string(uniform.name));
			if (it == uniformNames.end()) {
				uniform.index = uniforms.size();

				uniforms.push_back(uniform);
				uniformNames.insert(std::string(uniform.name));
			}
		}

		GLint attributeCount;
		glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attributeCount);
		for (int j = 0; j < attributeCount; j++)
		{
			GLsizei nameLength;
			GLint size;
			GLenum type;
			GLchar name[32];
			glGetActiveAttrib(program, (GLuint)j, bufSize, &nameLength, &size, &type, &name[0]);


			ShaderAttribute attribute;
			for (int k = 0; k < 32; k++) {
				attribute.name[k] = name[k];
			}

			std::set<std::string>::iterator it = attributeNames.find(std::string(attribute.name));
			if (it == attributeNames.end()) {
				attributes.push_back(attribute);
				attributeNames.insert(std::string(attribute.name));
			}
		}
	}
}

void Shader::use(int program)
{
	if (program == -1){
		return;
	}	

	activeProgram = program;
	glUseProgram(program);
}

void Shader::unuse()
{
	activeProgram = -1;
	glUseProgram(0);
}

void Shader::setVertexShader(const std::string vertexShader)
{
	this->vertexShader = vertexShader;
	allProgramsCompiled = false;
}

void Shader::setGeometryShader(const std::string geometryShader)
{
	this->geometryShader = geometryShader;
	allProgramsCompiled = false;
}

void Shader::setFragmentShader(const std::string fragmentShader)
{
	this->fragmentShader = fragmentShader;
	allProgramsCompiled = false;
}

void Shader::setUniformBlock(const std::string& blockName, int bindingPoint) const
{
	//set uniform block on all shader program
	for (size_t i = 0; i < programs.size(); i++) {
		GLuint blockIndex = glGetUniformBlockIndex(programs[i].handle, blockName.c_str());
		if (blockIndex != GL_INVALID_INDEX) {
			glUniformBlockBinding(programs[i].handle, blockIndex, bindingPoint);
		}
	}
}

int Shader::findUniformLocation(const std::string& name, int program) const
{
	return glGetUniformLocation(program, name.c_str());
}

int Shader::getProgramFromVariant(int variant) const
{
	for (size_t i = 0; i < programs.size(); i++) {
		if (programs[i].variant == variant) {
			return programs[i].handle;
		}
	}

	return -1;
}

std::vector<ShaderProgram> Shader::getPrograms() const
{
	return programs;
}

std::vector<ShaderUniform> Shader::getUniforms() const
{
	return uniforms;
}

 std::vector<ShaderAttribute> Shader::getAttributeNames() const
 {
	 return attributes;
 }

 std::string Shader::getVertexShader() const
 {
	 return vertexShader;
 }

 std::string Shader::getGeometryShader() const
 {
	 return geometryShader;
 }

 std::string Shader::getFragmentShader() const
 {
	 return fragmentShader;
 }

void Shader::setBool(const std::string& name, bool value) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, (int)value);
		}
	}
}

void Shader::setInt(const std::string& name, int value) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, value);
		}
	}
}

void Shader::setFloat(const std::string& name, float value) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1f(locationIndex, value);
		}
	}
}

void Shader::setVec2(const std::string& name, const glm::vec2 &vec) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform2fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec3(const std::string& name, const glm::vec3 &vec) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform3fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec4(const std::string& name, const glm::vec4 &vec) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform4fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setMat2(const std::string& name, const glm::mat2 &mat) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix2fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat3(const std::string& name, const glm::mat3 &mat) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix3fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat4(const std::string& name, const glm::mat4 &mat) const
{
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix4fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setBool(int nameLocation, bool value) const
{
	if(activeProgram != -1 && nameLocation != -1){
		glUniform1i(nameLocation, (int)value);
	}
}
void Shader::setInt(int nameLocation, int value) const
{
	if(activeProgram != -1 && nameLocation != -1){
		glUniform1i(nameLocation, value);
	}
}

void Shader::setFloat(int nameLocation, float value) const
{
	if(activeProgram != -1 && nameLocation != -1){
		glUniform1f(nameLocation, value);
	}
}

void Shader::setVec2(int nameLocation, const glm::vec2 &vec) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniform2fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec3(int nameLocation, const glm::vec3 &vec) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniform3fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec4(int nameLocation, const glm::vec4 &vec) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniform4fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setMat2(int nameLocation, const glm::mat2 &mat) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat3(int nameLocation, const glm::mat3 &mat) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat4(int nameLocation, const glm::mat4 &mat) const
{
	if (activeProgram != -1 && nameLocation != -1) {
		glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

bool Shader::getBool(const std::string& name) const
{
	int value = 0;
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(activeProgram, locationIndex, &value);
		}
	}

	return (bool)value;
}

int Shader::getInt(const std::string& name) const
{
	int value = 0;
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(activeProgram, locationIndex, &value);
		}
	}

	return value;
}

float Shader::getFloat(const std::string& name) const
{
	float value = 0.0f;
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformfv(activeProgram, locationIndex, &value);
		}
	}

	return value;
}

glm::vec2 Shader::getVec2(const std::string& name) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::vec2), &value[0]);
		}
	}

	return value;
}

glm::vec3 Shader::getVec3(const std::string& name) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::vec3), &value[0]);
		}
	}

	return value;
}

glm::vec4 Shader::getVec4(const std::string& name) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::vec4), &value[0]);
		}
	}

	return value;
}

glm::mat2 Shader::getMat2(const std::string& name) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::mat2), &value[0][0]);
		}
	}

	return value;
}

glm::mat3 Shader::getMat3(const std::string& name) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::mat3), &value[0][0]);
		}
	}

	return value;
}

glm::mat4 Shader::getMat4(const std::string& name) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (activeProgram != -1) {
		GLint locationIndex = glGetUniformLocation(activeProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(activeProgram, locationIndex, sizeof(glm::mat4), &value[0][0]);
		}
	}

	return value;
}

bool Shader::getBool(int nameLocation) const
{
	int value = 0;
	if (activeProgram != -1 && nameLocation != -1) {
		glGetUniformiv(activeProgram, nameLocation, &value);
	}

	return (bool)value;
}

int Shader::getInt(int nameLocation) const
{
	int value = 0;
	if (activeProgram != -1 && nameLocation != -1) {
		glGetUniformiv(activeProgram, nameLocation, &value);
	}

	return value;
}

float Shader::getFloat(int nameLocation) const
{
	float value = 0.0f;
	if (activeProgram != -1 && nameLocation != -1) {
		glGetUniformfv(activeProgram, nameLocation, &value);
	}

	return value;
}

glm::vec2 Shader::getVec2(int nameLocation) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::vec2), &value[0]);
	}

	return value;
}

glm::vec3 Shader::getVec3(int nameLocation) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::vec3), &value[0]);
	}

	return value;
}

glm::vec4 Shader::getVec4(int nameLocation) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::vec4), &value[0]);
	}

	return value;
}

glm::mat2 Shader::getMat2(int nameLocation) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::mat2), &value[0][0]);
	}

	return value;
}

glm::mat3 Shader::getMat3(int nameLocation) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::mat3), &value[0][0]);
	}

	return value;
}

glm::mat4 Shader::getMat4(int nameLocation) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (activeProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(activeProgram, nameLocation, sizeof(glm::mat4), &value[0][0]);
	}

	return value;
}