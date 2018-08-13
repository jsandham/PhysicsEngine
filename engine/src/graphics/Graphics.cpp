#include <iostream>
#include <GL/glew.h>

#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GLHandle.h"
#include "../../include/graphics/OpenGL.h"

using namespace PhysicsEngine;

void Graphics::initializeGraphicsAPI()
{
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK){
		std::cout << "Error: Could not initialize GLEW" << std::endl;
	}
}

void Graphics::readPixels(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glBindTexture(GL_TEXTURE_2D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glGetTextureImage(handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*numChannels, &rawTextureData[0]);
	
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::apply(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glBindTexture(GL_TEXTURE_2D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::generate(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glGenTextures(1, &handle);
	glBindTexture(GL_TEXTURE_2D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::destroy(Texture2D* texture)
{	
	GLuint handle = texture->getHandle().handle;

	glDeleteTextures(1, &handle);
}

void Graphics::bind(Texture2D* texture)
{
	GLuint handle = texture->getHandle().handle;

	glBindTexture(GL_TEXTURE_2D, handle);
}

void Graphics::unbind(Texture2D* texture)
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::active(Texture2D* texture, unsigned int slot)
{
	glActiveTexture(GL_TEXTURE0 + slot);
}

void Graphics::readPixels(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glBindTexture(GL_TEXTURE_3D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glGetTextureImage(handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*depth*numChannels, &rawTextureData[0]);
	
	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::apply(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glBindTexture(GL_TEXTURE_3D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::generate(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();
	GLuint handle = texture->getHandle().handle;

	glGenTextures(1, &handle);
	glBindTexture(GL_TEXTURE_3D, handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::destroy(Texture3D* texture)
{
	GLuint handle = texture->getHandle().handle;

	glDeleteTextures(1, &handle);
}

void Graphics::bind(Texture3D* texture)
{
	GLuint handle = texture->getHandle().handle;
	
	glBindTexture(GL_TEXTURE_2D, handle);
}

void Graphics::unbind(Texture3D* texture)
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::active(Texture3D* texture, unsigned int slot)
{
	//glActiveTexture(GL_TEXTURE0 + slot);
}

void Graphics::compile(Shader* shader)
{
	std::string vertexShader = shader->vertexShader;
	std::string geometryShader = shader->geometryShader;
	std::string fragmentShader = shader->fragmentShader;

	// GLuint vertexShader, fragmentShader, geometryShader;
	// GLchar infoLog[512];
	// success = 0;

	// // vertex shader
	// vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	// glCompileShader(vertexShader);
	// glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	// if (!success)
	// {
	//     glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
	//     Log::Error("Shader: Vertex shader compilation failed\n");
	//     return false;
	// }

	// // fragment shader
	// fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	// glCompileShader(fragmentShader);
	// glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	// if (!success)
	// {
	//     glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
	//     Log::Error("Shader: Fragment shader compilation failed\n");
	//     return false;
	// }

	// // geometry shader
	// if (!geometryShaderPath.empty()){
	// 	geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
	// 	glShaderSource(geometryShader, 1, &geometryShaderCode, NULL);
	// 	glCompileShader(geometryShader);
	// 	glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
	// 	if (!success)
	// 	{
	// 		glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
	// 		Log::Error("Shader: Geometry shader compilation failed\n");
	// 		return false;
	// 	}
	// }

	// // shader program
	// Program = glCreateProgram();
	// glAttachShader(Program, vertexShader);
	// glAttachShader(Program, fragmentShader);
	// if (!geometryShaderPath.empty()){
	// 	glAttachShader(Program, geometryShader);
	// }

	// glLinkProgram(Program);
	// glGetProgramiv(Program, GL_LINK_STATUS, &success);
	// if (!success) {
	//     glGetProgramInfoLog(Program, 512, NULL, infoLog);
	//     Log::Error("Shader: Shader program linking failed\n");
	//     return false;
	// }
	// glDeleteShader(vertexShader);
	// glDeleteShader(fragmentShader);
	// if (!geometryShaderPath.empty()){
	// 	glDeleteShader(geometryShader);
	// }

	// GLint count;
	// GLint size;
	// GLenum type;
	// GLchar name[16];
	// GLsizei length;

	// // fill vector of strings with attributes from shader program
	// glGetProgramiv(Program, GL_ACTIVE_ATTRIBUTES, &count);
	// for(int i = 0; i < count; i++){
	// 	glGetActiveAttrib(Program, (GLuint)i, 16, &length, &size, &type, name);
	// 	attributes.push_back(std::string(name));
	// 	attributeTypes.push_back(type);
	// 	//std::cout << "attribute type: " << type << std::endl;
	// 	//std::cout << "GL_FLOAT: " << GL_FLOAT << " GL_FLOAT_MAT4: " << GL_FLOAT_MAT4 << std::endl;
	// }

	// // fill vector of strings with uniforms from shader program
	// glGetProgramiv(Program, GL_ACTIVE_UNIFORMS, &count);
	// for(int i = 0; i < count; i++){
	// 	glGetActiveUniform(Program, (GLuint)i, 16, &length, &size, &type, name);
	// 	uniforms.push_back(std::string(name));
	// 	uniformTypes.push_back(type);
	// 	//std::cout << "uniform type: " << type << std::endl;
	// }

	// return true; 
}

void Graphics::bind(Shader* shader)
{
	if(shader->isCompiled()){
		std::cout << "Error: Must compile shader before using" << std::endl;
		return;
	}

	glUseProgram(shader->getHandle().handle);
}

void Graphics::unbind(Shader* shader)
{
	glUseProgram(0);
}



