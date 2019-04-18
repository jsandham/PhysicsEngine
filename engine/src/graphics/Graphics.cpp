#include <iostream>
#include <algorithm>
#include <GL/glew.h>

#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GraphicsState.h"

using namespace PhysicsEngine;


void DebugWindow::init()
{
	shader.vertexShader = Shader::windowVertexShader;
	shader.fragmentShader = Shader::windowFragmentShader;

	shader.compile();

	x = fmin(fmax(x, 0.0f), 1.0f);
	y = fmin(fmax(y, 0.0f), 1.0f);
	width = fmin(fmax(width, 0.0f), 1.0f);
	height = fmin(fmax(height, 0.0f), 1.0f);

	float x_ndc = 2.0f * x - 1.0f; 
	float y_ndc = 1.0f - 2.0f * y; 

	float width_ndc = 2.0f * width;
	float height_ndc = 2.0f * height;

	float vertices[18];
	float texCoords[12];

	vertices[0] = x_ndc; 
	vertices[1] = y_ndc;
	vertices[2] = 0.0f;

	vertices[3] = x_ndc; 
	vertices[4] = y_ndc - height_ndc; 
	vertices[5] = 0.0f; 

	vertices[6] = x_ndc + width_ndc; 
	vertices[7] = y_ndc;
	vertices[8] = 0.0f;  

	vertices[9] = x_ndc + width_ndc; 
	vertices[10] = y_ndc;  
	vertices[11] = 0.0f; 

	vertices[12] = x_ndc; 
	vertices[13] = y_ndc - height_ndc; 
	vertices[14] = 0.0f;  

	vertices[15] = x_ndc + width_ndc; 
	vertices[16] = y_ndc - height_ndc; 
	vertices[17] = 0.0f; 

	texCoords[0] = 0.0f;
	texCoords[1] = 1.0f;

	texCoords[2] = 0.0f;
	texCoords[3] = 0.0f;

	texCoords[4] = 1.0f;
	texCoords[5] = 1.0f;

	texCoords[6] = 1.0f;
	texCoords[7] = 1.0f;

	texCoords[8] = 0.0f;
	texCoords[9] = 0.0f;

	texCoords[10] = 1.0f;
	texCoords[11] = 0.0f;

	// for(int i = 0; i < 18; i++){
	// 	std::cout << vertices[i] << " ";
	// }

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glBufferData(GL_ARRAY_BUFFER, 18*sizeof(float), &(vertices[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glGenBuffers(1, &texCoordVBO);
	glBindBuffer(GL_ARRAY_BUFFER, texCoordVBO);
	glBufferData(GL_ARRAY_BUFFER, 12*sizeof(float), &(texCoords[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);
}

void PerformanceGraph::init()
{
	shader.vertexShader = Shader::graphVertexShader;
	shader.fragmentShader = Shader::graphFragmentShader;

	shader.compile();

	x = fmin(fmax(x, 0.0f), 1.0f);
	y = fmin(fmax(y, 0.0f), 1.0f);
	width = fmin(fmax(width, 0.0f), 1.0f);
	height = fmin(fmax(height, 0.0f), 1.0f);
	rangeMin = fmin(rangeMin, rangeMax);
	rangeMax = fmax(rangeMin, rangeMax);
	currentSample = 0.0f;
	numberOfSamples = std::max(2, numberOfSamples);

	samples.resize(18*numberOfSamples - 18);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, samples.size()*sizeof(float), &(samples[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);
}

void PerformanceGraph::add(float sample)
{
	float oldSample = currentSample;
	currentSample = fmin(fmax(sample, rangeMin), rangeMax);

	float dx = width / (numberOfSamples - 1);
	for(int i = 0; i < numberOfSamples - 2; i++){
		samples[18*i] = samples[18*(i+1)] - dx;
		samples[18*i + 1] = samples[18*(i+1) + 1];
		samples[18*i + 2] = samples[18*(i+1) + 2];

		samples[18*i + 3] = samples[18*(i+1) + 3] - dx;
		samples[18*i + 4] = samples[18*(i+1) + 4];
		samples[18*i + 5] = samples[18*(i+1) + 5];

		samples[18*i + 6] = samples[18*(i+1) + 6] - dx;
		samples[18*i + 7] = samples[18*(i+1) + 7];
		samples[18*i + 8] = samples[18*(i+1) + 8];

		samples[18*i + 9] = samples[18*(i+1) + 9] - dx;
		samples[18*i + 10] = samples[18*(i+1) + 10];
		samples[18*i + 11] = samples[18*(i+1) + 11];

		samples[18*i + 12] = samples[18*(i+1) + 12] - dx;
		samples[18*i + 13] = samples[18*(i+1) + 13];
		samples[18*i + 14] = samples[18*(i+1) + 14];

		samples[18*i + 15] = samples[18*(i+1) + 15] - dx;
		samples[18*i + 16] = samples[18*(i+1) + 16];
		samples[18*i + 17] = samples[18*(i+1) + 17];
	}

	float dz1 = 1.0f - (currentSample - rangeMin) / (rangeMax - rangeMin);
	float dz2 = 1.0f - (oldSample - rangeMin) / (rangeMax - rangeMin);

	float x_ndc = 2.0f * x - 1.0f;
	float y0_ndc = 1.0f - 2.0f * (y + height);
	float y1_ndc = 1.0f - 2.0f * (y + height * dz1);
	float y2_ndc = 1.0f - 2.0f * (y + height * dz2);

	samples[18*(numberOfSamples - 2)] = x_ndc + dx * (numberOfSamples - 2);
	samples[18*(numberOfSamples - 2) + 1] = y2_ndc;
	samples[18*(numberOfSamples - 2) + 2] = 0.0f;

	samples[18*(numberOfSamples - 2) + 3] = x_ndc + dx * (numberOfSamples - 2);
	samples[18*(numberOfSamples - 2) + 4] = y0_ndc;
	samples[18*(numberOfSamples - 2) + 5] = 0.0f;

	samples[18*(numberOfSamples - 2) + 6] = x_ndc + dx * (numberOfSamples - 1);
	samples[18*(numberOfSamples - 2) + 7] = y0_ndc;
	samples[18*(numberOfSamples - 2) + 8] = 0.0f;

	samples[18*(numberOfSamples - 2) + 9] = x_ndc + dx * (numberOfSamples - 2); 
	samples[18*(numberOfSamples - 2) + 10] = y2_ndc;
	samples[18*(numberOfSamples - 2) + 11] = 0.0f;

	samples[18*(numberOfSamples - 2) + 12] = x_ndc + dx * (numberOfSamples - 1);
	samples[18*(numberOfSamples - 2) + 13] = y0_ndc;
	samples[18*(numberOfSamples - 2) + 14] = 0.0f;

	samples[18*(numberOfSamples - 2) + 15] = x_ndc + dx * (numberOfSamples - 1);
	samples[18*(numberOfSamples - 2) + 16] = y1_ndc;
	samples[18*(numberOfSamples - 2) + 17] = 0.0f;

	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferSubData(GL_ARRAY_BUFFER, 0, samples.size()*sizeof(float), &(samples[0]));
}

void LineBuffer::init(std::vector<float> lines)
{
	shader.vertexShader = Shader::lineVertexShader;
	shader.fragmentShader = Shader::lineFragmentShader;

	shader.compile();

	size = lines.size();

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), &lines[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);
	glBindVertexArray(0);
}

void LineBuffer::update(std::vector<float> lines)
{
	if(lines.size() != size){
		std::cout << "Error: Cannot change buffer size after initialiation" << std::endl;
		return;
	}

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, lines.size() * sizeof(float), &lines[0]);
	glBindVertexArray(0);
}

// GLHandle Graphics::query;
// unsigned int Graphics::gpu_time;

// void Graphics::initializeGraphicsAPI()
// {
// 	glewExperimental = GL_TRUE;
// 	GLenum err = glewInit();
// 	if(err != GLEW_OK){
// 		std::cout << "Error: Could not initialize GLEW" << std::endl;
// 	}

// 	glGenQueries(1, &query.handle);
// }

void Graphics::checkError()
{
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
	}
}

GLenum Graphics::getTextureFormat(TextureFormat format)
{
	GLenum openglFormat = GL_DEPTH_COMPONENT;

	switch (format)
	{
	case Depth:
		openglFormat = GL_DEPTH_COMPONENT;
		break;
	case RG:
		openglFormat = GL_RG;
		break;
	case RGB:
		openglFormat = GL_RGB;
		break;
	case RGBA:
		openglFormat = GL_RGBA;
		break;
	default:
		std::cout << "OpengGL: Invalid texture format" << std::endl;
	}

	return openglFormat;
}

// void Graphics::enableBlend()
// {
// 	glEnable(GL_BLEND);
// }

// void Graphics::enableDepthTest()
// {
// 	glEnable(GL_DEPTH_TEST);
// }

// void Graphics::enableCubemaps()
// {
// 	glEnable(GL_TEXTURE_CUBE_MAP);
// }

// void Graphics::enablePoints()
// {
// 	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
// 	glEnable(GL_POINT_SPRITE);
// }

// void Graphics::setDepth(GLDepth depth)
// {
// 	GLenum d;
// 	switch(depth)
// 	{
// 		case GLDepth::Never:
// 			d = GL_NEVER;
// 			break;
// 		case GLDepth::Less:
// 			d = GL_LESS;
// 			break;
// 		case GLDepth::Equal:
// 			d = GL_EQUAL;
// 			break;
// 		case GLDepth::LEqual:
// 			d = GL_LEQUAL;
// 			break;
// 		case GLDepth::Greater:
// 			d = GL_GREATER;
// 			break;
// 		case GLDepth::NotEqual:
// 			d = GL_NOTEQUAL;
// 			break;
// 		case GLDepth::GEqual:
// 			d = GL_GEQUAL;
// 			break;
// 		default:
// 			d = GL_ALWAYS;
// 	}

// 	glDepthFunc(d);
// }

// void Graphics::setBlending(GLBlend src, GLBlend dest)
// {
// 	GLenum s, d;

// 	switch(src)
// 	{
// 		case GLBlend::Zero:
// 			s = GL_ZERO;
// 			break;
// 		case GLBlend::One:
// 			s = GL_ONE;
// 			break;
// 		default:
// 			s = GL_ONE;
// 	}

// 	switch(dest)
// 	{
// 		case GLBlend::Zero:
// 			d = GL_ZERO;
// 			break;
// 		case GLBlend::One:
// 			d = GL_ONE;
// 			break;
// 		default:
// 			d = GL_ONE;
// 	}

// 	glBlendFunc(s, d);
// 	glBlendEquation(GL_FUNC_ADD);
// }

// void Graphics::setViewport(int x, int y, int width, int height)
// {
// 	glViewport(x, y, width, height);
// }

// void Graphics::clearColorBuffer(glm::vec4 value)
// {
// 	glClearColor(value.x, value.y, value.z, value.w);
// 	glClear(GL_COLOR_BUFFER_BIT);
// }

// void Graphics::clearDepthBuffer(float value)
// {
// 	glClearDepth(value);
// 	glClear(GL_DEPTH_BUFFER_BIT);
// }

// void Graphics::beginGPUTimer()
// {
// 	glBeginQuery(GL_TIME_ELAPSED, query.handle);
// }

// int Graphics::endGPUTimer()
// {
// 	glEndQuery(GL_TIME_ELAPSED);
// 	glGetQueryObjectuiv(query.handle, GL_QUERY_RESULT, &gpu_time);

// 	return gpu_time;
// }

// void Graphics::generate(GLFramebuffer* framebuffer)
// {
// 	// int width, height, numChannels;
// 	// TextureFormat format;

// 	glGenFramebuffers(1, &(framebuffer->handle));
// 	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);

// 	if(framebuffer->colorBuffer != NULL){
// 		int width = framebuffer->colorBuffer->getWidth();
// 		int height = framebuffer->colorBuffer->getHeight();
// 		int numChannels = framebuffer->colorBuffer->getNumChannels();
// 		TextureFormat format = framebuffer->colorBuffer->getFormat();
// 		std::vector<unsigned char> rawTextureData = framebuffer->colorBuffer->getRawTextureData();

// 		glGenTextures(1, &(framebuffer->colorBuffer->handle.handle));
// 		glBindTexture(GL_TEXTURE_2D, framebuffer->colorBuffer->handle.handle);

// 		GLenum openglFormat = OpenGL::getTextureFormat(format);

// 		glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

// 		glBindTexture(GL_TEXTURE_2D, 0);

// 		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebuffer->colorBuffer->handle.handle, 0);
// 	}

// 	if(framebuffer->depthBuffer != NULL){
// 		int width = framebuffer->depthBuffer->getWidth();
// 		int height = framebuffer->depthBuffer->getHeight();
// 		int numChannels = framebuffer->depthBuffer->getNumChannels();
// 		TextureFormat format = framebuffer->depthBuffer->getFormat();
// 		std::vector<unsigned char> rawTextureData = framebuffer->depthBuffer->getRawTextureData();

// 		glGenTextures(1, &(framebuffer->depthBuffer->handle.handle));
// 		glBindTexture(GL_TEXTURE_2D, framebuffer->depthBuffer->handle.handle);

// 		GLenum openglFormat = OpenGL::getTextureFormat(format);

// 		std::cout << "depth opengl format: " << openglFormat << std::endl;

// 		glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
// 		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

// 		glBindTexture(GL_TEXTURE_2D, 0);

// 		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, framebuffer->depthBuffer->handle.handle, 0);
// 	}

// 	std::cout << "frame buffer handle: " << framebuffer->handle << " framebuffer buffer handle: " << framebuffer->colorBuffer->handle.handle << " framebuffer depth buffer handle: " << framebuffer->depthBuffer->handle.handle << std::endl;

// 	GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
// 	glDrawBuffers(1, DrawBuffers);

// 	if ((framebuffer->framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
// 		std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << framebuffer->framebufferStatus << std::endl;
// 	}
// 	glBindFramebuffer(GL_FRAMEBUFFER, 0);
// }

// void Graphics::bind(GLFramebuffer* framebuffer)
// {
// 	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->handle);
// }

// void Graphics::unbind(GLFramebuffer* framebuffer)
// {
// 	glBindFramebuffer(GL_FRAMEBUFFER, 0);
// }

void Graphics::readPixels(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glGetTextureImage(texture->handle.handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*numChannels, &rawTextureData[0]);
	
	texture->setRawTextureData(rawTextureData, width, height, format);

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

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_2D, 0);
}

// void Graphics::generate(Texture2D* texture)
// {
// 	int width = texture->getWidth();
// 	int height = texture->getHeight();
// 	int numChannels = texture->getNumChannels();
// 	TextureFormat format = texture->getFormat();
// 	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

// 	glGenTextures(1, &(texture->handle.handle));
// 	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 	glGenerateMipmap(GL_TEXTURE_2D);

// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
// 	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Graphics::destroy(Texture2D* texture)
// {	
// 	glDeleteTextures(1, &(texture->handle.handle));
// }

// void Graphics::bind(Texture2D* texture)
// {
// 	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);
// }

// void Graphics::unbind(Texture2D* texture)
// {
// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Graphics::active(Texture2D* texture, unsigned int slot)
// {
// 	glActiveTexture(GL_TEXTURE0 + slot);
// }

void Graphics::readPixels(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_3D, texture->handle.handle);

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glGetTextureImage(texture->handle.handle, 0, openglFormat, GL_UNSIGNED_BYTE, width*height*depth*numChannels, &rawTextureData[0]);

	//texture->setRawTextureData(rawTextureData, width, height, depth, format);
	
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

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_3D, 0);
}

// void Graphics::generate(Texture3D* texture)
// {
// 	int width = texture->getWidth();
// 	int height = texture->getHeight();
// 	int depth = texture->getDepth();
// 	int numChannels = texture->getNumChannels();
// 	TextureFormat format = texture->getFormat();
// 	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

// 	glGenTextures(1, &(texture->handle.handle));
// 	glBindTexture(GL_TEXTURE_3D, texture->handle.handle);

// 	GLenum openglFormat = OpenGL::getTextureFormat(format);

// 	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
// 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
// 	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

// 	glBindTexture(GL_TEXTURE_3D, 0);
// }

// void Graphics::destroy(Texture3D* texture)
// {
// 	glDeleteTextures(1, &(texture->handle.handle));
// }

// void Graphics::bind(Texture3D* texture)
// {
// 	glBindTexture(GL_TEXTURE_2D, texture->handle.handle);
// }

// void Graphics::unbind(Texture3D* texture)
// {
// 	glBindTexture(GL_TEXTURE_2D, 0);
// }

// void Graphics::active(Texture3D* texture, unsigned int slot)
// {
// 	//glActiveTexture(GL_TEXTURE0 + slot);
// }

void Graphics::readPixels(Cubemap* cubemap)
{

}

void Graphics::apply(Cubemap* cubemap)
{

}

// void Graphics::generate(Cubemap* cubemap)
// {

// }

// void Graphics::destroy(Cubemap* cubemap)
// {

// }

// void Graphics::bind(Cubemap* cubemap)
// {

// }

// void Graphics::unbind(Cubemap* cubemap)
// {

// }

// void Graphics::bind(World* world, Material* material, glm::mat4 model)
// {
// 	Shader* shader = world->getAsset<Shader>(material->shaderId);

// 	if(shader == NULL){
// 		std::cout << "Shader is NULL" << std::endl;
// 		return;
// 	}

// 	if(!shader->isCompiled()){
// 		std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
// 		return;
// 	}

// 	Graphics::use(shader);
// 	Graphics::setMat4(shader, "model", model);
// 	Graphics::setFloat(shader, "material.shininess", material->shininess);
// 	Graphics::setVec3(shader, "material.ambient", material->ambient);
// 	Graphics::setVec3(shader, "material.diffuse", material->diffuse);
// 	Graphics::setVec3(shader, "material.specular", material->specular);

// 	Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
// 	if(mainTexture != NULL){
// 		Graphics::setInt(shader, "material.mainTexture", 0);

// 		Graphics::active(mainTexture, 0);
// 		Graphics::bind(mainTexture);
// 	}

// 	Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
// 	if(normalMap != NULL){

// 		Graphics::setInt(shader, "material.normalMap", 1);

// 		Graphics::active(normalMap, 1);
// 		Graphics::bind(normalMap);
// 	}

// 	Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);
// 	if(specularMap != NULL){

// 		Graphics::setInt(shader, "material.specularMap", 2);

// 		Graphics::active(specularMap, 2);
// 		Graphics::bind(specularMap);
// 	}
// }

// void Graphics::unbind(Material* material)
// {

// }

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
	    std::cout << vertexShader << std::endl;
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
}

void Graphics::use(Shader* shader)
{
	if(!shader->isCompiled()){
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
	// else{
	// 	std::cout << "Error: set mat4 name: " << name << " location index: " << locationIndex << std::endl;
	// }
}

// void Graphics::setUniformBlockToBindingPoint(Shader* shader, std::string blockName, unsigned int bindingPoint)
// {
// 	GLuint blockIndex = glGetUniformBlockIndex(shader->program.handle, blockName.c_str()); 
// 	if (blockIndex != GL_INVALID_INDEX){
// 		glUniformBlockBinding(shader->program.handle, blockIndex, bindingPoint);
// 	}
// 	//else{
// 	//	std::cout << "error for block name: " << blockName << " block index: " << blockIndex << std::endl;
// 	//}
// }

// void Graphics::apply(Line* line)
// {
// 	glBindBuffer(GL_ARRAY_BUFFER, line->vertexVBO.handle);

// 	float vertices[6];

// 	vertices[0] = line->start.x;
// 	vertices[1] = line->start.y;
// 	vertices[2] = line->start.z;
// 	vertices[3] = line->end.x;
// 	vertices[4] = line->end.y;
// 	vertices[5] = line->end.z;

// 	glBufferSubData(GL_ARRAY_BUFFER, 0, 6*sizeof(float), &vertices[0]);
// }

// void Graphics::generate(Line* line)
// {
// 	glGenVertexArrays(1, &(line->lineVAO.handle));
// 	glBindVertexArray(line->lineVAO.handle);

// 	float vertices[6];

// 	vertices[0] = line->start.x;
// 	vertices[1] = line->start.y;
// 	vertices[2] = line->start.z;
// 	vertices[3] = line->end.x;
// 	vertices[4] = line->end.y;
// 	vertices[5] = line->end.z;

// 	glGenBuffers(1, &(line->vertexVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, line->vertexVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, 6*sizeof(float), &vertices[0], GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(0);
// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glBindVertexArray(0);
// }

// void Graphics::destroy(Line* line)
// {
// 	glDeleteVertexArrays(1, &(line->lineVAO.handle));
// 	glDeleteBuffers(1, &(line->vertexVBO.handle));
// }

// void Graphics::bind(Line* line)
// {
// 	glBindVertexArray(line->lineVAO.handle);
// }

// void Graphics::unbind(Line* line)
// {
// 	glBindVertexArray(0);
// }

// void Graphics::draw(Line* line)
// {
// 	glDrawArrays(GL_LINES, 0, 2);
// }

// void Graphics::apply(Mesh* mesh)
// {

// }

// void Graphics::generate(Mesh* mesh)
// {
// 	glGenVertexArrays(1, &(mesh->meshVAO.handle));
// 	glBindVertexArray(mesh->meshVAO.handle);

// 	glGenBuffers(1, &(mesh->vertexVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size()*sizeof(float), &(mesh->vertices[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(0);
// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glGenBuffers(1, &(mesh->normalVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, mesh->normalVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, mesh->normals.size()*sizeof(float), &(mesh->normals[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(1);
// 	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glGenBuffers(1, &(mesh->texCoordVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, mesh->texCoordVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, mesh->texCoords.size()*sizeof(float), &(mesh->texCoords[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(2);
// 	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

// 	glBindVertexArray(0);
// }

// void Graphics::destroy(Mesh* mesh)
// {
// 	glDeleteVertexArrays(1, &(mesh->meshVAO.handle));
// 	glDeleteBuffers(1, &(mesh->vertexVBO.handle));
// 	glDeleteBuffers(1, &(mesh->normalVBO.handle));
// 	glDeleteBuffers(1, &(mesh->texCoordVBO.handle));
// }

// void Graphics::bind(Mesh* mesh)
// {
// 	glBindVertexArray(mesh->meshVAO.handle);
// }

// void Graphics::unbind(Mesh* mesh)
// {
// 	glBindVertexArray(0);
// }

// void Graphics::draw(Mesh* mesh)
// {
// 	glDrawArrays(GL_TRIANGLES, 0, (GLsizei)mesh->vertices.size() / 3);
// }

// void Graphics::apply(Boids* boids)
// {

// }

// void Graphics::generate(Boids* boids)
// {
// 	glGenBuffers(1, &boids->handle.handle);
// 	glBindBuffer(GL_ARRAY_BUFFER, boids->handle.handle);
// 	glBufferData(GL_ARRAY_BUFFER, boids->numBoids * sizeof(glm::mat4), &modelMatrices[0], GL_DYNAMIC_DRAW);

// 	unsigned int VAO = rock.meshes[i].VAO;
//     glBindVertexArray(VAO);
//     // vertex Attributes
//     glEnableVertexAttribArray(3); 
//     glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)0);
//     glEnableVertexAttribArray(4); 
//     glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4)));
//     glEnableVertexAttribArray(5); 
//     glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(2 * sizeof(glm::vec4)));
//     glEnableVertexAttribArray(6); 
//     glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(3 * sizeof(glm::vec4)));

//     glVertexAttribDivisor(3, 1);
//     glVertexAttribDivisor(4, 1);
//     glVertexAttribDivisor(5, 1);
//     glVertexAttribDivisor(6, 1);

//     glBindVertexArray(0);
// }

// void Graphics::destroy(Boids* boids)
// {

// }

// void Graphics::bind(Boids* boids)
// {

// }

// void Graphics::unbind(Boids* boids)
// {

// }

// void Graphics::draw(Boids* boids)
// {

// }

// void Graphics::apply(PerformanceGraph* graph)
// {
// 	glBindBuffer(GL_ARRAY_BUFFER, graph->vertexVBO.handle);

// 	glBufferSubData(GL_ARRAY_BUFFER, 0, graph->vertices.size()*sizeof(float), &(graph->vertices[0]));
// }

// void Graphics::generate(PerformanceGraph* graph)
// {
// 	glGenVertexArrays(1, &(graph->graphVAO.handle));
// 	glBindVertexArray(graph->graphVAO.handle);

// 	glGenBuffers(1, &(graph->vertexVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, graph->vertexVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, graph->vertices.size()*sizeof(float), &(graph->vertices[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(0);
// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glBindVertexArray(0);
// }

// void Graphics::destroy(PerformanceGraph* graph)
// {
// 	glDeleteVertexArrays(1, &(graph->graphVAO.handle));
// 	glDeleteBuffers(1, &(graph->vertexVBO.handle));
// }

// void Graphics::bind(PerformanceGraph* graph)
// {
// 	glBindVertexArray(graph->graphVAO.handle);
// }

// void Graphics::unbind(PerformanceGraph* graph)
// {
// 	glBindVertexArray(0);
// }

// void Graphics::draw(PerformanceGraph* graph)
// {
// 	glDrawArrays(GL_TRIANGLES, 0, (GLsizei)graph->vertices.size() / 3);
// }

// void Graphics::apply(DebugWindow* window)
// {

// }

// void Graphics::generate(DebugWindow* window)
// {
// 	glGenVertexArrays(1, &(window->windowVAO.handle));
// 	glBindVertexArray(window->windowVAO.handle);

// 	glGenBuffers(1, &(window->vertexVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, window->vertexVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, window->vertices.size()*sizeof(float), &(window->vertices[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(0);
// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glGenBuffers(1, &(window->texCoordVBO.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, window->texCoordVBO.handle);
// 	glBufferData(GL_ARRAY_BUFFER, window->texCoords.size()*sizeof(float), &(window->texCoords[0]), GL_STATIC_DRAW);
// 	glEnableVertexAttribArray(1);
// 	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

// 	std::cout << "window vertices size: " << window->vertices.size() << " window texcoords size: " << window->texCoords.size() << std::endl;
// 	std::cout << "texCoords[0]: " << window->texCoords[0] << " texCoords[1]: " << window->texCoords[1] << " texCoords[2]: " << window->texCoords[2] << " texCoords[3]: " << window->texCoords[3] << " texCoords[4]: " << window->texCoords[4] << std::endl;

// 	glBindVertexArray(0);
// }

// void Graphics::destroy(DebugWindow* window)
// {
// 	glDeleteVertexArrays(1, &(window->windowVAO.handle));
// 	glDeleteBuffers(1, &(window->vertexVBO.handle));
// 	glDeleteBuffers(1, &(window->texCoordVBO.handle));
// }

// void Graphics::bind(DebugWindow* window)
// {
// 	glBindVertexArray(window->windowVAO.handle);
// }

// void Graphics::unbind(DebugWindow* window)
// {
// 	glBindVertexArray(0);
// }

// void Graphics::draw(DebugWindow* window)
// {
// 	glDrawArrays(GL_TRIANGLES, 0, (GLsizei)window->vertices.size() / 3);
// }

// void Graphics::apply(SlabNode* node)
// {
// 	glBindBuffer(GL_ARRAY_BUFFER, node->vbo.handle); 

// 	glBufferSubData(GL_ARRAY_BUFFER, 0, node->count*sizeof(float), &(node->buffer[0]));
// }

// void Graphics::generate(SlabNode* node)
// {
// 	glGenVertexArrays(1, &(node->vao.handle));
// 	glBindVertexArray(node->vao.handle);

// 	glGenBuffers(1, &(node->vbo.handle));
// 	glBindBuffer(GL_ARRAY_BUFFER, node->vbo.handle);
// 	glBufferData(GL_ARRAY_BUFFER, node->size*sizeof(float), &(node->buffer[0]), GL_DYNAMIC_DRAW);
// 	glEnableVertexAttribArray(0);
// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

// 	glBindVertexArray(0);
// }

// void Graphics::destroy(SlabNode* node)
// {
// 	glDeleteVertexArrays(1, &(node->vao.handle));
// 	glDeleteBuffers(1, &(node->vbo.handle));
// }

// void Graphics::bind(SlabNode* node)
// {
// 	glBindVertexArray(node->vao.handle);
// }

// void Graphics::unbind(SlabNode* node)
// {
// 	glBindVertexArray(0);
// }

// void Graphics::draw(SlabNode* node)
// {
// 	glDrawArrays(GL_LINES, 0, (GLsizei)node->count / 3);
// }

// void Graphics::generate(GLCamera* state)
// {
// 	glGenBuffers(1, &(state->handle.handle));
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// 	glBufferData(GL_UNIFORM_BUFFER, 144, NULL, GL_DYNAMIC_DRAW);
// 	glBindBufferRange(GL_UNIFORM_BUFFER, 0, state->handle.handle, 0, 144);
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::destroy(GLCamera* state)
// {
// 	glDeleteBuffers(1, &(state->handle.handle));
// }

// void Graphics::bind(GLCamera* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// }

// void Graphics::unbind(GLCamera* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::setProjectionMatrix(GLCamera* state, glm::mat4 projection)
// {
// 	state->projection = projection;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 0, 64, glm::value_ptr(state->projection));
// }

// void Graphics::setViewMatrix(GLCamera* state, glm::mat4 view)
// {
// 	state->view = view;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 64, 64, glm::value_ptr(state->view));
// }

// void Graphics::setCameraPosition(GLCamera* state, glm::vec3 position)
// {
// 	state->cameraPos = position;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 128, 16, glm::value_ptr(state->cameraPos));
// }

// void Graphics::generate(GLShadow* state)
// {
// 	glGenBuffers(1, &(state->handle.handle));
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// 	glBufferData(GL_UNIFORM_BUFFER, 736, NULL, GL_DYNAMIC_DRAW);
// 	glBindBufferRange(GL_UNIFORM_BUFFER, 4, state->handle.handle, 0, 736);
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::destroy(GLShadow* state)
// {
// 	glDeleteBuffers(1, &(state->handle.handle));
// }

// void Graphics::bind(GLShadow* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// }

// void Graphics::unbind(GLShadow* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::setLightProjectionMatrix(GLShadow* state, glm::mat4 projection, int index)
// {
// 	//glBufferSubData(GL_UNIFORM_BUFFER, 64 * index, 64, data);
// }

// void Graphics::setLightViewMatrix(GLShadow* state, glm::mat4 view, int index)
// {
// 	//glBufferSubData(GL_UNIFORM_BUFFER, 320 + 64 * index, 64, data);
// }

// void Graphics::setCascadeEnd(GLShadow* state, float cascadeEnd, int index)
// {
// 	//glBufferSubData(GL_UNIFORM_BUFFER, 640 + 16 * index, 16, data);
// }

// void Graphics::setFarPlane(GLShadow* state, float farPlane)
// {
// 	//glBufferSubData(GL_UNIFORM_BUFFER, 720, 16, data);
// }

// void Graphics::generate(GLDirectionalLight* state)
// {
// 	glGenBuffers(1, &(state->handle.handle));
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// 	glBufferData(GL_UNIFORM_BUFFER, 64, NULL, GL_DYNAMIC_DRAW);
// 	glBindBufferRange(GL_UNIFORM_BUFFER, 1, state->handle.handle, 0, 64);
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::destroy(GLDirectionalLight* state)
// {
// 	glDeleteBuffers(1, &(state->handle.handle));
// }

// void Graphics::bind(GLDirectionalLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// }

// void Graphics::unbind(GLDirectionalLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::setDirLightDirection(GLDirectionalLight* state, glm::vec3 direction)
// {
// 	state->dirLightDirection = direction;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->dirLightDirection));
// }

// void Graphics::setDirLightAmbient(GLDirectionalLight* state, glm::vec3 ambient)
// {
// 	state->dirLightAmbient = ambient;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->dirLightAmbient));
// }

// void Graphics::setDirLightDiffuse(GLDirectionalLight* state, glm::vec3 diffuse)
// {
// 	state->dirLightDiffuse = diffuse;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->dirLightDiffuse));
// }

// void Graphics::setDirLightSpecular(GLDirectionalLight* state, glm::vec3 specular)
// {
// 	state->dirLightSpecular = specular;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->dirLightSpecular));
// }

// void Graphics::generate(GLSpotLight* state)
// {
// 	glGenBuffers(1, &(state->handle.handle));
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// 	glBufferData(GL_UNIFORM_BUFFER, 100, NULL, GL_DYNAMIC_DRAW);
// 	glBindBufferRange(GL_UNIFORM_BUFFER, 2, state->handle.handle, 0, 100);
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::destroy(GLSpotLight* state)
// {
// 	glDeleteBuffers(1, &(state->handle.handle));
// }

// void Graphics::bind(GLSpotLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// }

// void Graphics::unbind(GLSpotLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::setSpotLightPosition(GLSpotLight* state, glm::vec3 position)
// {
// 	state->spotLightPosition = position;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->spotLightPosition));
// }

// void Graphics::setSpotLightDirection(GLSpotLight* state, glm::vec3 direction)
// {
// 	state->spotLightDirection = direction;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->spotLightDirection));
// }

// void Graphics::setSpotLightAmbient(GLSpotLight* state, glm::vec3 ambient)
// {
// 	state->spotLightAmbient = ambient;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->spotLightAmbient));
// }

// void Graphics::setSpotLightDiffuse(GLSpotLight* state, glm::vec3 diffuse)
// {
// 	state->spotLightDiffuse = diffuse;
	
// 	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->spotLightDiffuse));
// }

// void Graphics::setSpotLightSpecular(GLSpotLight* state, glm::vec3 specular)
// {
// 	state->spotLightSpecular = specular;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 64, 16, glm::value_ptr(state->spotLightSpecular));
// }

// void Graphics::setSpotLightConstant(GLSpotLight* state, float constant)
// {
// 	state->spotLightConstant = constant;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 80, 4, &(state->spotLightConstant));
// }

// void Graphics::setSpotLightLinear(GLSpotLight* state, float linear)
// {
// 	state->spotLightLinear = linear;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 84, 4, &(state->spotLightLinear));
// }

// void Graphics::setSpotLightQuadratic(GLSpotLight* state, float quadratic)
// {
// 	state->spotLightQuadratic = quadratic;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 88, 4, &(state->spotLightQuadratic));
// }

// void Graphics::setSpotLightCutoff(GLSpotLight* state, float cutoff)
// {
// 	state->spotLightCutoff = cutoff;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 92, 4, &(state->spotLightCutoff));
// }

// void Graphics::setSpotLightOuterCutoff(GLSpotLight* state, float cutoff)
// {
// 	state->spotLightOuterCutoff = cutoff;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 96, 4, &(state->spotLightOuterCutoff));
// }

// void Graphics::generate(GLPointLight* state)
// {
// 	glGenBuffers(1, &(state->handle.handle));
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// 	glBufferData(GL_UNIFORM_BUFFER, 76, NULL, GL_DYNAMIC_DRAW);
// 	glBindBufferRange(GL_UNIFORM_BUFFER, 3, state->handle.handle, 0, 76);
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::destroy(GLPointLight* state)
// {
// 	glDeleteBuffers(1, &(state->handle.handle));
// }

// void Graphics::bind(GLPointLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, state->handle.handle);
// }

// void Graphics::unbind(GLPointLight* state)
// {
// 	glBindBuffer(GL_UNIFORM_BUFFER, 0);
// }

// void Graphics::setPointLightPosition(GLPointLight* state, glm::vec3 position)
// {
// 	state->pointLightPosition = position;
	
// 	glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, glm::value_ptr(state->pointLightPosition));
// }

// void Graphics::setPointLightAmbient(GLPointLight* state, glm::vec3 ambient)
// {
// 	state->pointLightAmbient = ambient;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 16, 16, glm::value_ptr(state->pointLightAmbient));
// }

// void Graphics::setPointLightDiffuse(GLPointLight* state, glm::vec3 diffuse)
// {
// 	state->pointLightDiffuse = diffuse;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 32, 16, glm::value_ptr(state->pointLightDiffuse));
// }

// void Graphics::setPointLightSpecular(GLPointLight* state, glm::vec3 specular)
// {
// 	state->pointLightSpecular = specular;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 48, 16, glm::value_ptr(state->pointLightSpecular));
// }

// void Graphics::setPointLightConstant(GLPointLight* state, float constant)
// {
// 	state->pointLightConstant = constant;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 64, 4, &state->pointLightConstant);
// }

// void Graphics::setPointLightLinear(GLPointLight* state, float linear)
// {
// 	state->pointLightLinear = linear;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 68, 4, &state->pointLightLinear);
// }

// void Graphics::setPointLightQuadratic(GLPointLight* state, float quadratic)
// {
// 	state->pointLightQuadratic = quadratic;

// 	glBufferSubData(GL_UNIFORM_BUFFER, 72, 4, &state->pointLightQuadratic);
// }


void Graphics::render(World* world, Material* material, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query) 
{
	Shader* shader = world->getAsset<Shader>(material->shaderId);

	if(material == NULL){
		std::cout << "Material is NULL" << std::endl;
		return;
	}

	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
		return;
	}

	Graphics::use(shader);
	Graphics::setMat4(shader, "model", model);
	Graphics::setFloat(shader, "material.shininess", material->shininess);
	Graphics::setVec3(shader, "material.ambient", material->ambient);
	Graphics::setVec3(shader, "material.diffuse", material->diffuse);
	Graphics::setVec3(shader, "material.specular", material->specular);

	Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
	if(mainTexture != NULL){
		Graphics::setInt(shader, "material.mainTexture", 0);

		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, mainTexture->handle.handle);
	}

	Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
	if(normalMap != NULL){

		Graphics::setInt(shader, "material.normalMap", 1);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, normalMap->handle.handle);
	}

	Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);
	if(specularMap != NULL){

		Graphics::setInt(shader, "material.specularMap", 2);

		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, specularMap->handle.handle);
	}

	glBindVertexArray(vao);

	if(world->debug && query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
	}

	glDrawArrays(GL_TRIANGLES, 0, numVertices);

	if(world->debug && query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->queryId, 
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime;
		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

		query->totalElapsedTime += elapsedTime;
		query->numDrawCalls++;
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
	}
}

void Graphics::render(World* world, Shader* shader, Texture2D* texture, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query)
{
	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
		return;
	}

	Graphics::use(shader);
	Graphics::setMat4(shader, "model", model);

	if(texture != NULL){
		Graphics::setInt(shader, "texture0", 0);

		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, texture->handle.handle);
	}

	glBindVertexArray(vao);

	if(world->debug && query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
	}

	glDrawArrays(GL_TRIANGLES, 0, numVertices);

	if(world->debug && query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->queryId, 
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime;
		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

		query->totalElapsedTime += elapsedTime;
		query->numDrawCalls++;
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
	}
}

void Graphics::render(World* world, Shader* shader, glm::mat4 model, GLuint vao, GLenum mode, int numVertices, GraphicsQuery* query)
{
	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
		return;
	}

	Graphics::use(shader);
	Graphics::setMat4(shader, "model", model);

	glBindVertexArray(vao);

	if(world->debug && query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
	}

	glDrawArrays(mode, 0, numVertices);

	if(world->debug && query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->queryId, 
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime;
		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

		query->totalElapsedTime += elapsedTime;
		query->numDrawCalls++;
	}

	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "Error: Renderer failed with error code: " << error << std::endl;;
	}
}

void Graphics::renderText(World* world, Camera* camera, Font* font, std::string text, float x, float y, float scale, glm::vec3 color)
{
	if(!font->shader.isCompiled()){
		std::cout << "Shader " << font->shader.assetId.toString() << " has not been compiled." << std::endl;
		return;
	}

	Graphics::use(&font->shader);
	Graphics::setMat4(&font->shader, "projection", glm::ortho(0.0f, (float)camera->width, 0.0f, (float)camera->height));
	Graphics::setVec3(&font->shader, "textColor", color);

	glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(font->vao.handle);

    // Iterate through all characters
    std::string::const_iterator it;
    for (it = text.begin(); it != text.end(); it++) 
    {
        Character ch = font->getCharacter(*it);//Characters[*it];

        GLfloat xpos = x + ch.bearing.x * scale;
        GLfloat ypos = y - (ch.size.y - ch.bearing.y) * scale;

        GLfloat w = ch.size.x * scale;
        GLfloat h = ch.size.y * scale;
        // Update VBO for each character
        GLfloat vertices[6][4] = {
            { xpos,     ypos + h,   0.0, 0.0 },            
            { xpos,     ypos,       0.0, 1.0 },
            { xpos + w, ypos,       1.0, 1.0 },

            { xpos,     ypos + h,   0.0, 0.0 },
            { xpos + w, ypos,       1.0, 1.0 },
            { xpos + w, ypos + h,   1.0, 0.0 }           
        };
        // Render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.glyphId.handle);

        // Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, font->vbo.handle);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}