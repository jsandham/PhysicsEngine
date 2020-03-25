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
	mVertexShader = "";
	mFragmentShader = "";
	mGeometryShader = "";

	mAssetId = Guid::INVALID;

	mAllProgramsCompiled = false;
	mActiveProgram = -1;
}

Shader::Shader(std::vector<char> data)
{
	deserialize(data);

	mAllProgramsCompiled = false;
	mActiveProgram = -1;
}

Shader::~Shader()
{

}

std::vector<char> Shader::serialize() const
{
	return serialize(mAssetId);
}

std::vector<char> Shader::serialize(Guid assetId) const
{
	ShaderHeader header;
	header.mShaderId = assetId;
	header.mVertexShaderSize = mVertexShader.length();
	header.mGeometryShaderSize = mGeometryShader.length();
	header.mFragmentShaderSize = mFragmentShader.length();
	header.mNumberOfShaderUniforms = mUniforms.size();

	size_t numberOfBytes = sizeof(ShaderHeader) +
		sizeof(char) * mVertexShader.length() +
		sizeof(char) * mFragmentShader.length() +
		sizeof(char) * mGeometryShader.length() +
		sizeof(ShaderUniform) * mUniforms.size();

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(ShaderHeader);
	size_t start3 = start2 + sizeof(char) * mVertexShader.length();
	size_t start4 = start3 + sizeof(char) * mGeometryShader.length();

	memcpy(&data[start1], &header, sizeof(ShaderHeader));
	memcpy(&data[start2], mVertexShader.c_str(), sizeof(char) * mVertexShader.length());
	memcpy(&data[start3], mGeometryShader.c_str(), sizeof(char) * mGeometryShader.length());
	memcpy(&data[start4], mFragmentShader.c_str(), sizeof(char) * mFragmentShader.length());

	return data;
}

void Shader::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(ShaderHeader);

	ShaderHeader* header = reinterpret_cast<ShaderHeader*>(&data[start1]);

	mAssetId = header->mShaderId;

	size_t vertexShaderSize = header->mVertexShaderSize;
	size_t geometryShaderSize = header->mGeometryShaderSize;
	size_t fragmentShaderSize = header->mFragmentShaderSize;

	std::vector<char>::iterator start = data.begin();
	std::vector<char>::iterator end = data.begin();
	start += start2;
	end += start2 + vertexShaderSize;

	mVertexShader = std::string(start, end);

	start +=vertexShaderSize;
	end += geometryShaderSize;

	mGeometryShader = std::string(start, end);

	start += geometryShaderSize;
	end += fragmentShaderSize;

	mFragmentShader = std::string(start, end);
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
	return mAllProgramsCompiled;
}

bool Shader::contains(int variant) const
{
	for (size_t i = 0; i < mPrograms.size(); i++) {
		if (mPrograms[i].mVariant == variant) {
			return true;
		}
	}

	return false;
}

void Shader::add(int variant)
{
	bool variantFound = false;
	for (size_t i = 0; i < mPrograms.size(); i++) {
		if (mPrograms[i].mVariant == variant) {
			variantFound = true;
			break;
		}
	}

	if (!variantFound) {
		ShaderProgram program;
		program.mVersion = ShaderVersion::GL430;
		program.mCompiled = false;
		program.mVariant = variant;
		program.mHandle = 0;

		mPrograms.push_back(program);

		mAllProgramsCompiled = false;
	}
}

void Shader::remove(int variant)
{
	int index = -1;
	for (size_t i = 0; i < mPrograms.size(); i++) {
		if (mPrograms[i].mVariant == variant) {
			index = (int)i;
			break;
		}
	}

	if (index != -1) {
		mPrograms.erase(mPrograms.begin() + index);
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
		if (mVertexShader.find(keywords[i]) != std::string::npos || 
			mGeometryShader.find(keywords[i]) != std::string::npos || 
			mFragmentShader.find(keywords[i]) != std::string::npos) {
			
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
	for (size_t i = 0; i < mPrograms.size(); i++) {
		std::string version;
		if(mPrograms[i].mVersion == ShaderVersion::GL330) { 
			version = "#version 330 core\n"; 
		}
		else{ 
			version = "#version 430 core\n";
		}

		std::string defines;
		if (mPrograms[i].mVariant & ShaderVariant::Directional) { defines += "#define DIRECTIONALLIGHT\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::Spot) { defines += "#define SPOTLIGHT\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::Point) { defines += "#define POINTLIGHT\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::HardShadows) { defines += "#define HARDSHADOWS\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::SoftShadows) { defines += "#define SOFTSHADOWS\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::SSAO) { defines += "#define SSAO\n"; }
		if (mPrograms[i].mVariant & ShaderVariant::Cascade) { defines += "#define CASCADE\n"; }

		std::string preProcessedVertexShader = version + defines + mVertexShader;
		std::string preProcessedGeometryShader = version + defines + mGeometryShader;
		std::string preProcessedFragmentShader = version + defines + mFragmentShader;

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
		if (!mGeometryShader.empty()) {
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
		mPrograms[i].mHandle = glCreateProgram();
		glAttachShader(mPrograms[i].mHandle, vertexShaderObj);
		glAttachShader(mPrograms[i].mHandle, fragmentShaderObj);
		if (geometryShaderObj != 0) {
			glAttachShader(mPrograms[i].mHandle, geometryShaderObj);
		}

		glLinkProgram(mPrograms[i].mHandle);
		glGetProgramiv(mPrograms[i].mHandle, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(mPrograms[i].mHandle, 512, NULL, infoLog);
			std::string message = "Shader: Shader program linking failed\n";
			Log::error(message.c_str());
			return;
		}
		glDeleteShader(vertexShaderObj);
		glDeleteShader(fragmentShaderObj);
		if (!mGeometryShader.empty()) {
			glDeleteShader(geometryShaderObj);
		}

		mPrograms[i].mCompiled = true;
	}

	mAllProgramsCompiled = true;

	// find all uniforms and attributes in shader across all variants
	std::set<std::string> uniformNames;
	for (size_t i = 0; i < mUniforms.size(); i++) {
		uniformNames.insert(std::string(mUniforms[i].mName));
	}
	std::set<std::string> attributeNames;
	for (size_t i = 0; i < mAttributes.size(); i++) {
		attributeNames.insert(std::string(mAttributes[i].mName));
	}

	const GLsizei bufSize = 32; // maximum name length

	// run through all variants and find all uniforms/attributes (and add to sets of known uniforms/attributes if new)
	for (size_t i = 0; i < mPrograms.size(); i++) {
		GLuint program = mPrograms[i].mHandle;

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
			uniform.mNameLength = (size_t)nameLength;
			uniform.mSize = (size_t)size;

			memset(uniform.mName, '\0', 32);
			memset(uniform.mShortName, '\0', 32);
			memset(uniform.mBlockName, '\0', 32);

			int indexOfBlockChar = -1;
			for (int k = 0; k < nameLength; k++) {
				uniform.mName[k] = name[k];
				if (name[k] == '.') {
					indexOfBlockChar = k;
				}
			}

			uniform.mShortName[0] = '\0';
			for (int k = indexOfBlockChar + 1; k < nameLength; k++) {
				uniform.mShortName[k - indexOfBlockChar - 1] = name[k];
			}

			uniform.mBlockName[0] = '\0';
			for (int k = 0; k < indexOfBlockChar; k++) {
				uniform.mBlockName[k] = name[k];
			}

			uniform.mType = type;
			uniform.mVariant = mPrograms[i].mVariant;
			uniform.mLocation = findUniformLocation(std::string(uniform.mName), program);

			// only add uniform if it wasnt already in array
			std::set<std::string>::iterator it = uniformNames.find(std::string(uniform.mName));
			if (it == uniformNames.end()) {
				uniform.mIndex = mUniforms.size();

				mUniforms.push_back(uniform);
				uniformNames.insert(std::string(uniform.mName));
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
				attribute.mName[k] = name[k];
			}

			std::set<std::string>::iterator it = attributeNames.find(std::string(attribute.mName));
			if (it == attributeNames.end()) {
				mAttributes.push_back(attribute);
				attributeNames.insert(std::string(attribute.mName));
			}
		}
	}
}

void Shader::use(int program)
{
	if (program == -1){
		return;
	}	

	mActiveProgram = program;
	glUseProgram(program);
}

void Shader::unuse()
{
	mActiveProgram = -1;
	glUseProgram(0);
}

void Shader::setVertexShader(const std::string vertexShader)
{
	mVertexShader = vertexShader;
	mAllProgramsCompiled = false;
}

void Shader::setGeometryShader(const std::string geometryShader)
{
	mGeometryShader = geometryShader;
	mAllProgramsCompiled = false;
}

void Shader::setFragmentShader(const std::string fragmentShader)
{
	mFragmentShader = fragmentShader;
	mAllProgramsCompiled = false;
}

void Shader::setUniformBlock(const std::string& blockName, int bindingPoint) const
{
	//set uniform block on all shader program
	for (size_t i = 0; i < mPrograms.size(); i++) {
		GLuint blockIndex = glGetUniformBlockIndex(mPrograms[i].mHandle, blockName.c_str());
		if (blockIndex != GL_INVALID_INDEX) {
			glUniformBlockBinding(mPrograms[i].mHandle, blockIndex, bindingPoint);
		}
	}
}

int Shader::findUniformLocation(const std::string& name, int program) const
{
	return glGetUniformLocation(program, name.c_str());
}

int Shader::getProgramFromVariant(int variant) const
{
	for (size_t i = 0; i < mPrograms.size(); i++) {
		if (mPrograms[i].mVariant == variant) {
			return mPrograms[i].mHandle;
		}
	}

	return -1;
}

std::vector<ShaderProgram> Shader::getPrograms() const
{
	return mPrograms;
}

std::vector<ShaderUniform> Shader::getUniforms() const
{
	return mUniforms;
}

 std::vector<ShaderAttribute> Shader::getAttributeNames() const
 {
	 return mAttributes;
 }

 std::string Shader::getVertexShader() const
 {
	 return mVertexShader;
 }

 std::string Shader::getGeometryShader() const
 {
	 return mGeometryShader;
 }

 std::string Shader::getFragmentShader() const
 {
	 return mFragmentShader;
 }

void Shader::setBool(const std::string& name, bool value) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, (int)value);
		}
	}
}

void Shader::setInt(const std::string& name, int value) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1i(locationIndex, value);
		}
	}
}

void Shader::setFloat(const std::string& name, float value) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform1f(locationIndex, value);
		}
	}
}

void Shader::setVec2(const std::string& name, const glm::vec2 &vec) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform2fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec3(const std::string& name, const glm::vec3 &vec) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform3fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setVec4(const std::string& name, const glm::vec4 &vec) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniform4fv(locationIndex, 1, &vec[0]);
		}
	}
}

void Shader::setMat2(const std::string& name, const glm::mat2 &mat) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix2fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat3(const std::string& name, const glm::mat3 &mat) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix3fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setMat4(const std::string& name, const glm::mat4 &mat) const
{
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glUniformMatrix4fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
		}
	}
}

void Shader::setBool(int nameLocation, bool value) const
{
	if(mActiveProgram != -1 && nameLocation != -1){
		glUniform1i(nameLocation, (int)value);
	}
}
void Shader::setInt(int nameLocation, int value) const
{
	if(mActiveProgram != -1 && nameLocation != -1){
		glUniform1i(nameLocation, value);
	}
}

void Shader::setFloat(int nameLocation, float value) const
{
	if(mActiveProgram != -1 && nameLocation != -1){
		glUniform1f(nameLocation, value);
	}
}

void Shader::setVec2(int nameLocation, const glm::vec2 &vec) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniform2fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec3(int nameLocation, const glm::vec3 &vec) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniform3fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setVec4(int nameLocation, const glm::vec4 &vec) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniform4fv(nameLocation, 1, &vec[0]);
	}
}

void Shader::setMat2(int nameLocation, const glm::mat2 &mat) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniformMatrix2fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat3(int nameLocation, const glm::mat3 &mat) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniformMatrix3fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setMat4(int nameLocation, const glm::mat4 &mat) const
{
	if (mActiveProgram != -1 && nameLocation != -1) {
		glUniformMatrix4fv(nameLocation, 1, GL_FALSE, &mat[0][0]);
	}
}

bool Shader::getBool(const std::string& name) const
{
	int value = 0;
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(mActiveProgram, locationIndex, &value);
		}
	}

	return (bool)value;
}

int Shader::getInt(const std::string& name) const
{
	int value = 0;
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformiv(mActiveProgram, locationIndex, &value);
		}
	}

	return value;
}

float Shader::getFloat(const std::string& name) const
{
	float value = 0.0f;
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetUniformfv(mActiveProgram, locationIndex, &value);
		}
	}

	return value;
}

glm::vec2 Shader::getVec2(const std::string& name) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::vec2), &value[0]);
		}
	}

	return value;
}

glm::vec3 Shader::getVec3(const std::string& name) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::vec3), &value[0]);
		}
	}

	return value;
}

glm::vec4 Shader::getVec4(const std::string& name) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::vec4), &value[0]);
		}
	}

	return value;
}

glm::mat2 Shader::getMat2(const std::string& name) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::mat2), &value[0][0]);
		}
	}

	return value;
}

glm::mat3 Shader::getMat3(const std::string& name) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::mat3), &value[0][0]);
		}
	}

	return value;
}

glm::mat4 Shader::getMat4(const std::string& name) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (mActiveProgram != -1) {
		GLint locationIndex = glGetUniformLocation(mActiveProgram, name.c_str());
		if (locationIndex != -1) {
			glGetnUniformfv(mActiveProgram, locationIndex, sizeof(glm::mat4), &value[0][0]);
		}
	}

	return value;
}

bool Shader::getBool(int nameLocation) const
{
	int value = 0;
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetUniformiv(mActiveProgram, nameLocation, &value);
	}

	return (bool)value;
}

int Shader::getInt(int nameLocation) const
{
	int value = 0;
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetUniformiv(mActiveProgram, nameLocation, &value);
	}

	return value;
}

float Shader::getFloat(int nameLocation) const
{
	float value = 0.0f;
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetUniformfv(mActiveProgram, nameLocation, &value);
	}

	return value;
}

glm::vec2 Shader::getVec2(int nameLocation) const
{
	glm::vec2 value = glm::vec2(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::vec2), &value[0]);
	}

	return value;
}

glm::vec3 Shader::getVec3(int nameLocation) const
{
	glm::vec3 value = glm::vec3(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::vec3), &value[0]);
	}

	return value;
}

glm::vec4 Shader::getVec4(int nameLocation) const
{
	glm::vec4 value = glm::vec4(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::vec4), &value[0]);
	}

	return value;
}

glm::mat2 Shader::getMat2(int nameLocation) const
{
	glm::mat2 value = glm::mat2(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::mat2), &value[0][0]);
	}

	return value;
}

glm::mat3 Shader::getMat3(int nameLocation) const
{
	glm::mat3 value = glm::mat3(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::mat3), &value[0][0]);
	}

	return value;
}

glm::mat4 Shader::getMat4(int nameLocation) const
{
	glm::mat4 value = glm::mat4(0.0f);
	if (mActiveProgram != -1 && nameLocation != -1) {
		glGetnUniformfv(mActiveProgram, nameLocation, sizeof(glm::mat4), &value[0][0]);
	}

	return value;
}