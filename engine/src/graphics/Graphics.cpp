#include <iostream>
#include <GL/glew.h>

#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GLHandle.h"
#include "../../include/graphics/OpenGL.h"
#include "../../include/graphics/GLState.h"

using namespace PhysicsEngine;

void Graphics::initializeGraphicsAPI()
{
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK){
		std::cout << "Error: Could not initialize GLEW" << std::endl;
	}
}

void Graphics::checkError()
{
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
	}
}

void Graphics::enableDepthTest()
{
	glEnable(GL_DEPTH_TEST);
}

void Graphics::enableCubemaps()
{
	glEnable(GL_TEXTURE_CUBE_MAP);
}

void Graphics::enablePoints()
{
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
}

void Graphics::setViewport(int x, int y, int width, int height)
{
	glViewport(x, y, width, height);
}

void Graphics::clearColorBuffer(glm::vec4 value)
{
	glClearColor(value.x, value.y, value.z, value.w);
	glClear(GL_COLOR_BUFFER_BIT);
}

void Graphics::clearDepthBuffer(float value)
{
	glClearDepth(value);
	glClear(GL_DEPTH_BUFFER_BIT);
}

void Graphics::readPixels(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glGetTextureImage(texture->handle.handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*numChannels, &rawTextureData[0]);
	
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::apply(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

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

	glGenTextures(1, &(texture->handle.handle));
	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

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
	glDeleteTextures(1, &(texture->handle.handle));
}

void Graphics::bind(Texture2D* texture)
{
	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);
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

	glBindTexture(GL_TEXTURE_3D, texture->handle.handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glGetTextureImage(texture->handle.handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*depth*numChannels, &rawTextureData[0]);
	
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

	glBindTexture(GL_TEXTURE_3D, texture->handle.handle);

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

	glGenTextures(1, &(texture->handle.handle));
	glBindTexture(GL_TEXTURE_3D, texture->handle.handle);

	GLenum openglFormat = OpenGL::getTextureFormat(format);

	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::destroy(Texture3D* texture)
{
	glDeleteTextures(1, &(texture->handle.handle));
}

void Graphics::bind(Texture3D* texture)
{
	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);
}

void Graphics::unbind(Texture3D* texture)
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::active(Texture3D* texture, unsigned int slot)
{
	//glActiveTexture(GL_TEXTURE0 + slot);
}

void Graphics::readPixels(Cubemap* cubemap)
{

}

void Graphics::apply(Cubemap* cubemap)
{

}

void Graphics::generate(Cubemap* cubemap)
{

}

void Graphics::destroy(Cubemap* cubemap)
{

}

void Graphics::bind(Cubemap* cubemap)
{

}

void Graphics::unbind(Cubemap* cubemap)
{

}

void Graphics::compile(Shader* shader)
{
	const GLchar* vertexShader = shader->vertexShader.c_str();
	const GLchar* geometryShader = shader->geometryShader.c_str();
	const GLchar* fragmentShader = shader->fragmentShader.c_str();

	GLuint vertexShaderObj = 0;
	GLuint fragmentShaderObj = 0;
	GLuint geometryShaderObj = 0;
	GLchar infoLog[512];
	GLint success = 0;

	// vertex shader
	vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShaderObj, 1, &vertexShader, NULL);
	glCompileShader(vertexShaderObj);
	glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &success);
	if (!success)
	{
	    glGetShaderInfoLog(vertexShaderObj, 512, NULL, infoLog);
	    std::cout << "Shader: Vertex shader compilation failed" << std::endl;
	    return;
	}

	// fragment shader
	fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShaderObj, 1, &fragmentShader, NULL);
	glCompileShader(fragmentShaderObj);
	glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &success);
	if (!success)
	{
	    glGetShaderInfoLog(fragmentShaderObj, 512, NULL, infoLog);
	    std::cout << "Shader: Fragment shader compilation failed" << std::endl;
	    return;
	}

	// geometry shader
	if (!shader->geometryShader.empty()){
		geometryShaderObj = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometryShaderObj, 1, &geometryShader, NULL);
		glCompileShader(geometryShaderObj);
		glGetShaderiv(geometryShaderObj, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(geometryShaderObj, 512, NULL, infoLog);
			std::cout << "Shader: Geometry shader compilation failed" << std::endl;
			return;
		}
	}

	// shader program
	shader->program.handle = glCreateProgram();
	glAttachShader(shader->program.handle, vertexShaderObj);
	glAttachShader(shader->program.handle, fragmentShaderObj);
	if (geometryShaderObj != 0){
		glAttachShader(shader->program.handle, geometryShaderObj);
	}

	glLinkProgram(shader->program.handle);
	glGetProgramiv(shader->program.handle, GL_LINK_STATUS, &success);
	if (!success) {
	    glGetProgramInfoLog(shader->program.handle, 512, NULL, infoLog);
	    std::cout << "Shader: Shader program linking failed" << std::endl;
	    return;
	}
	glDeleteShader(vertexShaderObj);
	glDeleteShader(fragmentShaderObj);
	if (!shader->geometryShader.empty()){
		glDeleteShader(geometryShaderObj);
	}

	shader->programCompiled = (success != 0);

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
}

void Graphics::use(Shader* shader)
{
	if(shader->isCompiled()){
		std::cout << "Error: Must compile shader before using" << std::endl;
		return;
	}

	glUseProgram(shader->program.handle);
}

void Graphics::unuse(Shader* shader)
{
	glUseProgram(0);
}

void Graphics::setBool(Shader* shader, std::string name, bool value)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform1i(locationIndex, (int)value);
	}
}

void Graphics::setInt(Shader* shader, std::string name, int value)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform1i(locationIndex, (int)value);
	}
}

void Graphics::setFloat(Shader* shader, std::string name, float value)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform1f(locationIndex, value);
	}
}

void Graphics::setVec2(Shader* shader, std::string name, glm::vec2 &vec)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform2fv(locationIndex, 1, &vec[0]);
	}
}

void Graphics::setVec3(Shader* shader, std::string name, glm::vec3 &vec)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform3fv(locationIndex, 1, &vec[0]);
	}
}

void Graphics::setVec4(Shader* shader, std::string name, glm::vec4 &vec)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniform4fv(locationIndex, 1, &vec[0]);
	}
}

void Graphics::setMat2(Shader* shader, std::string name, glm::mat2 &mat)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniformMatrix2fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
	}
}

void Graphics::setMat3(Shader* shader, std::string name, glm::mat3 &mat)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniformMatrix3fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
	}
}

void Graphics::setMat4(Shader* shader, std::string name, glm::mat4 &mat)
{
	GLint locationIndex = glGetUniformLocation(shader->program.handle, name.c_str());
	if (locationIndex != -1){
		glUniformMatrix4fv(locationIndex, 1, GL_FALSE, &mat[0][0]);
	}
}

void Graphics::setUniformBlockToBindingPoint(Shader* shader, std::string blockName, unsigned int bindingPoint)
{
	GLuint blockIndex = glGetUniformBlockIndex(shader->program.handle, blockName.c_str()); 
	if (blockIndex != GL_INVALID_INDEX){
		glUniformBlockBinding(shader->program.handle, blockIndex, bindingPoint);
	}
}

void Graphics::apply(Mesh* mesh)
{

}

void Graphics::generate(Mesh* mesh)
{
	glGenVertexArrays(1, &(mesh->meshVAO.handle));
	glBindVertexArray(mesh->meshVAO.handle);

	glGenBuffers(1, &(mesh->vertexVBO.handle));
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexVBO.handle);
	glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size()*sizeof(float), &(mesh->vertices[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);


	glGenBuffers(1, &(mesh->normalVBO.handle));
	glBindBuffer(GL_ARRAY_BUFFER, mesh->normalVBO.handle);
	glBufferData(GL_ARRAY_BUFFER, mesh->normals.size()*sizeof(float), &(mesh->normals[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glGenBuffers(1, &(mesh->texCoordVBO.handle));
	glBindBuffer(GL_ARRAY_BUFFER, mesh->texCoordVBO.handle);
	glBufferData(GL_ARRAY_BUFFER, mesh->texCoords.size()*sizeof(float), &(mesh->texCoords[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);
}

void Graphics::destroy(Mesh* mesh)
{
	glDeleteVertexArrays(1, &(mesh->meshVAO.handle));
	glDeleteBuffers(1, &(mesh->vertexVBO.handle));
	glDeleteBuffers(1, &(mesh->normalVBO.handle));
	glDeleteBuffers(1, &(mesh->texCoordVBO.handle));

}

void Graphics::bind(Mesh* mesh)
{
	glBindVertexArray(mesh->meshVAO.handle);
}

void Graphics::unbind(Mesh* mesh)
{
	glBindVertexArray(0);
}

void Graphics::generate(GLCamera* state)
{
	glGenBuffers(1, &(state->handle.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 144);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::destroy(GLCamera* state)
{
	glDeleteBuffers(1, &(state->handle.handle));
}

void Graphics::bind(GLCamera* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
}

void Graphics::unbind(GLCamera* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::setProjectionMatrix(GLCamera* state, glm::mat4 projection)
{
	state->projection = projection;

	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state->projection));
}

void Graphics::setViewMatrix(GLCamera* state, glm::mat4 view)
{
	state->view = view;

	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state->view));
}

void Graphics::setCameraPosition(GLCamera* state, glm::vec3 position)
{
	state->cameraPos = position;

	glBufferSubData(GL_UNIFORM_BUFFER, 128, 16, glm::value_ptr(state->cameraPos));
}

void Graphics::generate(GLShadow* state)
{
	glGenBuffers(1, &(state->handle.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
	glBufferData(GL_UNIFORM_BUFFER, 736, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 736);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::destroy(GLShadow* state)
{
	glDeleteBuffers(1, &(state->handle.handle));
}

void Graphics::bind(GLShadow* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
}

void Graphics::unbind(GLShadow* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::setLightProjectionMatrix(GLShadow* state, glm::mat4 projection, int index)
{
	//glBufferSubData(GL_UNIFORM_BUFFER, 64 * index, 64, data);
}

void Graphics::setLightViewMatrix(GLShadow* state, glm::mat4 view, int index)
{
	//glBufferSubData(GL_UNIFORM_BUFFER, 320 + 64 * index, 64, data);
}

void Graphics::setCascadeEnd(GLShadow* state, float cascadeEnd, int index)
{
	//glBufferSubData(GL_UNIFORM_BUFFER, 640 + 16 * index, 16, data);
}

void Graphics::setFarPlane(GLShadow* state, float farPlane)
{
	//glBufferSubData(GL_UNIFORM_BUFFER, 720, 16, data);
}

void Graphics::generate(GLDirectionalLight* state)
{
	glGenBuffers(1, &(state->handle.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
	glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 64);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::destroy(GLDirectionalLight* state)
{
	glDeleteBuffers(1, &(state->handle.handle));
}

void Graphics::bind(GLDirectionalLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
}

void Graphics::unbind(GLDirectionalLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::setDirLightDirection(GLDirectionalLight* state, glm::vec3 direction)
{
	state->dirLightDirection = direction;

	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->dirLightDirection));
}

void Graphics::setDirLightAmbient(GLDirectionalLight* state, glm::vec3 ambient)
{
	state->dirLightAmbient = ambient;

	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->dirLightAmbient));
}

void Graphics::setDirLightDiffuse(GLDirectionalLight* state, glm::vec3 diffuse)
{
	state->dirLightDiffuse = diffuse;

	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->dirLightDiffuse));
}

void Graphics::setDirLightSpecular(GLDirectionalLight* state, glm::vec3 specular)
{
	state->dirLightSpecular = specular;

	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->dirLightSpecular));
}

void Graphics::generate(GLSpotLight* state)
{
	glGenBuffers(1, &(state->handle.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
	glBufferData(GL_UNIFORM_BUFFER, 100, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 100);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::destroy(GLSpotLight* state)
{
	glDeleteBuffers(1, &(state->handle.handle));
}

void Graphics::bind(GLSpotLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
}

void Graphics::unbind(GLSpotLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::setSpotLightPosition(GLSpotLight* state, glm::vec3 position)
{
	state->spotLightPosition = position;

	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->spotLightPosition));
}

void Graphics::setSpotLightDirection(GLSpotLight* state, glm::vec3 direction)
{
	state->spotLightDirection = direction;

	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->spotLightDirection));
}

void Graphics::setSpotLightAmbient(GLSpotLight* state, glm::vec3 ambient)
{
	state->spotLightAmbient = ambient;

	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->spotLightAmbient));
}

void Graphics::setSpotLightDiffuse(GLSpotLight* state, glm::vec3 diffuse)
{
	state->spotLightDiffuse = diffuse;
	
	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->spotLightDiffuse));
}

void Graphics::setSpotLightSpecular(GLSpotLight* state, glm::vec3 specular)
{
	state->spotLightSpecular = specular;

	glBufferSubData(GL_UNIFORM_BUFFER, 64, 16, glm::value_ptr(state->spotLightSpecular));
}

void Graphics::setSpotLightConstant(GLSpotLight* state, float constant)
{
	state->spotLightConstant = constant;

	glBufferSubData(GL_UNIFORM_BUFFER, 80, 4, &(state->spotLightConstant));
}

void Graphics::setSpotLightLinear(GLSpotLight* state, float linear)
{
	state->spotLightLinear = linear;

	glBufferSubData(GL_UNIFORM_BUFFER, 84, 4, &(state->spotLightLinear));
}

void Graphics::setSpotLightQuadratic(GLSpotLight* state, float quadratic)
{
	state->spotLightQuadratic = quadratic;

	glBufferSubData(GL_UNIFORM_BUFFER, 88, 4, &(state->spotLightQuadratic));
}

void Graphics::setSpotLightCutoff(GLSpotLight* state, float cutoff)
{
	state->spotLightCutoff = cutoff;

	glBufferSubData(GL_UNIFORM_BUFFER, 92, 4, &(state->spotLightCutoff));
}

void Graphics::setSpotLightOuterCutoff(GLSpotLight* state, float cutoff)
{
	state->spotLightOuterCutoff = cutoff;

	glBufferSubData(GL_UNIFORM_BUFFER, 96, 4, &(state->spotLightOuterCutoff));
}

void Graphics::generate(GLPointLight* state)
{
	glGenBuffers(1, &(state->handle.handle));
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
	glBufferData(GL_UNIFORM_BUFFER, 76, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 76);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::destroy(GLPointLight* state)
{
	glDeleteBuffers(1, &(state->handle.handle));
}

void Graphics::bind(GLPointLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
}

void Graphics::unbind(GLPointLight* state)
{
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void Graphics::setPointLightPosition(GLPointLight* state, glm::vec3 position)
{
	state->pointLightPosition = position;
	
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->pointLightPosition));
}

void Graphics::setPointLightAmbient(GLPointLight* state, glm::vec3 ambient)
{
	state->pointLightAmbient = ambient;

	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->pointLightAmbient));
}

void Graphics::setPointLightDiffuse(GLPointLight* state, glm::vec3 diffuse)
{
	state->pointLightDiffuse = diffuse;

	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->pointLightDiffuse));
}

void Graphics::setPointLightSpecular(GLPointLight* state, glm::vec3 specular)
{
	state->pointLightSpecular = specular;

	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->pointLightSpecular));
}

void Graphics::setPointLightConstant(GLPointLight* state, float constant)
{
	state->pointLightConstant = constant;

	glBufferSubData(GL_UNIFORM_BUFFER, 64, 4, &state->pointLightConstant);
}

void Graphics::setPointLightLinear(GLPointLight* state, float linear)
{
	state->pointLightLinear = linear;

	glBufferSubData(GL_UNIFORM_BUFFER, 68, 4, &state->pointLightLinear);
}

void Graphics::setPointLightQuadratic(GLPointLight* state, float quadratic)
{
	state->pointLightQuadratic = quadratic;

	glBufferSubData(GL_UNIFORM_BUFFER, 72, 4, &state->pointLightQuadratic);
}