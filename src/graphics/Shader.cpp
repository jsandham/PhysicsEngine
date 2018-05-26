#include <iostream>
#include "Shader.h"

#include "../core/Log.h"

using namespace PhysicsEngine;


Shader::Shader()
{
}


Shader::Shader(std::string vertexShaderPath, std::string fragmentShaderPath, std::string geometryShaderPath)
{
	this->vertexShaderPath = vertexShaderPath;
	this->fragmentShaderPath = fragmentShaderPath;
	this->geometryShaderPath = geometryShaderPath;
}

Shader::~Shader()
{
	//std::cout << vertexShaderPath << " shader destructure called" << std::endl;
}

void Shader::bind()
{
	if(!success){
		Log::Error("Shader: Must compile shader being using");
		return;
	}
	glUseProgram(this->Program);
}

void Shader::unbind()
{
	glUseProgram(0);
}


bool Shader::compile()
{
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK){
		Log::Error("Shader: Could not initialize GLEW");
		return false;
	}

	std::string vertexCode;
	std::string fragmentCode;
	std::string geometryCode;

	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;
	std::ifstream geometryShaderFile;

	vertexShaderFile.exceptions(std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::badbit);
	geometryShaderFile.exceptions(std::ifstream::badbit);

	try{
		vertexShaderFile.open(vertexShaderPath.c_str());
		std::stringstream vertexShaderStream;
		vertexShaderStream << vertexShaderFile.rdbuf();
		vertexShaderFile.close();
		vertexCode = vertexShaderStream.str();

		fragmentShaderFile.open(fragmentShaderPath.c_str());
		std::stringstream fragmentShaderStream;
		fragmentShaderStream << fragmentShaderFile.rdbuf();
		fragmentShaderFile.close();
		fragmentCode = fragmentShaderStream.str();

		if (!geometryShaderPath.empty()){
			geometryShaderFile.open(geometryShaderPath.c_str());
			std::stringstream geometryShaderStream;
			geometryShaderStream << geometryShaderFile.rdbuf();
			geometryShaderFile.close();
			geometryCode = geometryShaderStream.str();
		}
	}
	catch(std::ifstream::failure e){
		Log::Error("Shader: Shader file not successfully read");
		return false;
	}

	const GLchar* vertexShaderCode = vertexCode.c_str();
	const GLchar* fragmentShaderCode = fragmentCode.c_str();
	const GLchar* geometryShaderCode = geometryCode.c_str();

	GLuint vertexShader, fragmentShader, geometryShader;
	GLchar infoLog[512];
	success = 0;

	// vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        Log::Error("Shader: Vertex shader compilation failed\n");
        return false;
    }

    // fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        Log::Error("Shader: Fragment shader compilation failed\n");
        return false;
    }

	// geometry shader
	if (!geometryShaderPath.empty()){
		geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometryShader, 1, &geometryShaderCode, NULL);
		glCompileShader(geometryShader);
		glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
			Log::Error("Shader: Geometry shader compilation failed\n");
			return false;
		}
	}

    // shader program
	Program = glCreateProgram();
	glAttachShader(Program, vertexShader);
	glAttachShader(Program, fragmentShader);
	if (!geometryShaderPath.empty()){
		glAttachShader(Program, geometryShader);
	}

	glLinkProgram(Program);
	glGetProgramiv(Program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(Program, 512, NULL, infoLog);
        Log::Error("Shader: Shader program linking failed\n");
        return false;
    }
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	if (!geometryShaderPath.empty()){
		glDeleteShader(geometryShader);
	}

	GLint count;
	GLint size;
	GLenum type;
	GLchar name[16];
	GLsizei length;

	// fill vector of strings with attributes from shader program
	glGetProgramiv(Program, GL_ACTIVE_ATTRIBUTES, &count);
	for(int i = 0; i < count; i++){
		glGetActiveAttrib(Program, (GLuint)i, 16, &length, &size, &type, name);
		attributes.push_back(std::string(name));
		attributeTypes.push_back(type);
		//std::cout << "attribute type: " << type << std::endl;
		//std::cout << "GL_FLOAT: " << GL_FLOAT << " GL_FLOAT_MAT4: " << GL_FLOAT_MAT4 << std::endl;
	}

	// fill vector of strings with uniforms from shader program
	glGetProgramiv(Program, GL_ACTIVE_UNIFORMS, &count);
	for(int i = 0; i < count; i++){
		glGetActiveUniform(Program, (GLuint)i, 16, &length, &size, &type, name);
		uniforms.push_back(std::string(name));
		uniformTypes.push_back(type);
		//std::cout << "uniform type: " << type << std::endl;
	}

	return true; 
}

bool Shader::isCompiled()
{
	return (bool)success;
}

bool Shader::compile(std::string vertexShaderPath, std::string fragmentShaderPath, std::string geometryShaderPath)
{
	this->vertexShaderPath = vertexShaderPath;
	this->fragmentShaderPath = fragmentShaderPath;
	this->geometryShaderPath = geometryShaderPath;

	return compile();
}


GLint Shader::getAttributeLocation(const char *name) 
{
	return glGetAttribLocation(Program, name);
	//if(attribute == -1){
	// 	std::cout << "Could not bind attribute " << name << std::endl;
	//}
	//return attribute;
}

GLint Shader::getUniformLocation(const char *name) 
{
	return glGetUniformLocation(Program, name);
	 //if(uniform == -1){
	 //	std::cout << "Could not bind uniform " << name << std::endl;
	 //}
	//return uniform;
}

GLuint Shader::getUniformBlockIndex(const char *name)
{
	return glGetUniformBlockIndex(Program, name);
}



std::vector<std::string> Shader::getAttributes()
{
	return attributes;
}


std::vector<std::string> Shader::getUniforms()
{
	return uniforms;
}


std::vector<GLenum> Shader::getAttributeTypes()
{
	return attributeTypes;
}


std::vector<GLenum> Shader::getUniformTypes()
{
	return uniformTypes;
}


std::string Shader::getVertexShaderName()
{
	return vertexShaderPath;
}


std::string Shader::getFragmentShaderName()
{
	return fragmentShaderPath;
}


std::string Shader::getGeometryShaderName()
{
	return geometryShaderPath;
}


void Shader::setBool(std::string name, bool value)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform1i(locationIndex, (int)value);
	}
}


void Shader::setInt(std::string name, int value)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform1i(getUniformLocation(name.c_str()), value);
	}
}


void Shader::setFloat(std::string name, float value)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform1f(getUniformLocation(name.c_str()), value);
	}
}


void Shader::setVec2(std::string name, glm::vec2 &vec)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform2fv(getUniformLocation(name.c_str()), 1, &vec[0]);
	}
}


void Shader::setVec3(std::string name, glm::vec3 &vec) 
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform3fv(getUniformLocation(name.c_str()), 1, &vec[0]);
	}
}


void Shader::setVec4(std::string name, glm::vec4 &vec)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniform4fv(getUniformLocation(name.c_str()), 1, &vec[0]);
	}
}


void Shader::setMat2(std::string name, glm::mat2 &mat)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniformMatrix2fv(getUniformLocation(name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}
}


void Shader::setMat3(std::string name, glm::mat3 &mat)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniformMatrix3fv(getUniformLocation(name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}
}


void Shader::setMat4(std::string name, glm::mat4 &mat)
{
	GLint locationIndex = getUniformLocation(name.c_str());
	if (locationIndex != -1){
		glUniformMatrix4fv(getUniformLocation(name.c_str()), 1, GL_FALSE, &mat[0][0]);
	}
}

void Shader::setUniformBlock(std::string name, GLuint bindingPoint)
{
	GLuint blockIndex = getUniformBlockIndex(name.c_str());
	if (blockIndex != GL_INVALID_INDEX){
		glUniformBlockBinding(Program, getUniformBlockIndex(name.c_str()), bindingPoint);
	}
}