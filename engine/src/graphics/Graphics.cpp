#include <iostream>
#include <algorithm>
#include <random>
#include <GL/glew.h>

#include "../../include/core/Log.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GraphicsState.h"

using namespace PhysicsEngine;

void Graphics::checkError()
{
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::string errorStr;
		switch( error ) {
			case GL_INVALID_ENUM:
				errorStr = "Error: An unacceptable value is specified for an enumerated argument";
				break;
			case GL_INVALID_VALUE:
				errorStr = "Error: A numeric argument is out of range";
				break;
			case GL_INVALID_OPERATION:
				errorStr = "Error: The specified operation is not allowed in the current state";
				break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:
				errorStr = "Error: The framebuffer object is not complete";
				break;
			case GL_OUT_OF_MEMORY:
				errorStr = "Error: There is not enough money left to execute the command";
				break;
			case GL_STACK_UNDERFLOW:
				errorStr = "Error: An attempt has been made to perform an operation that would cause an internal stack to underflow";
				break;
			case GL_STACK_OVERFLOW:
				errorStr = "Error: An attempt has been made to perform an operation that would cause an internal stack to overflow";
				break;
			default:
				errorStr = "Error: Unknown error";
				break;
		}

		std::string errorMessage = errorStr + "(" + std::to_string(error) + ")\n";
		Log::error(errorMessage.c_str());
	}
}

void Graphics::checkFrambufferError()
{
	GLenum framebufferStatus = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
	if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE){
		std::string errorStr;
		switch(framebufferStatus)
		{
			case GL_FRAMEBUFFER_UNDEFINED:
				errorStr = "Error: The current FBO binding is 0 but no default framebuffer exists";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
				errorStr = "Error: One of the buffers enabled for rendering is incomplete";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
				errorStr = "Error: No buffers are attached to the FBO and it is not configured for rendering without attachments";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
				errorStr = "Error: Not all attachments enabled via glDrawBuffers exists in framebuffer";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
				errorStr = "Error: Not all buffers specified via glReadBuffer exists in framebuffer";
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED:
				errorStr = "Error: The combination of internal buffer formats is unsupported";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
				errorStr = "Error: The number of samples for each attachment is not the same";
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
				errorStr = "Error: Not all color attachments are layered textures or bound to the same target";
				break;
			default:
				errorStr = "Error: Unknown framebuffer status error";
				break;
		}

		std::string errorMessage = errorStr + "(" + std::to_string(framebufferStatus) + ")\n";
		Log::error(errorMessage.c_str());
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
		Log::error("OpengGL: Invalid texture format\n");
	}

	return openglFormat;
}

void Graphics::create(Camera* camera,
					  GLuint* mainFBO,
					  GLuint* colorTex,
					  GLuint* depthTex,
					  GLuint* colorPickingFBO,
					  GLuint* colorPickingTex,
					  GLuint* colorPickingDepthTex,
					  GLuint* geometryFBO,
					  GLuint* positionTex,
					  GLuint* normalTex,
					  GLuint* ssaoFBO,
					  GLuint* ssaoColorTex,
					  GLuint* ssaoNoiseTex,
					  glm::vec3* ssaoSamples,
				      bool* created)
{
	// generate main camera fbo (color + depth)
	glGenFramebuffers(1, mainFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, *mainFBO);

	glGenTextures(1, colorTex);
	glBindTexture(GL_TEXTURE_2D, *colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, depthTex);
	glBindTexture(GL_TEXTURE_2D, *depthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *depthTex, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int mainAttachments[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, mainAttachments);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// generate color picking fbo (color + depth)
	glGenFramebuffers(1, colorPickingFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, *colorPickingFBO);

	glGenTextures(1, colorPickingTex);
	glBindTexture(GL_TEXTURE_2D, *colorPickingTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, colorPickingDepthTex);
	glBindTexture(GL_TEXTURE_2D, *colorPickingDepthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *colorPickingTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *colorPickingDepthTex, 0);

	// - tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int colorPickingAttachments[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, colorPickingAttachments);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// generate geometry fbo
	glGenFramebuffers(1, geometryFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, *geometryFBO);

	glGenTextures(1, positionTex);
	glBindTexture(GL_TEXTURE_2D, *positionTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glGenTextures(1, normalTex);
	glBindTexture(GL_TEXTURE_2D, *normalTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *positionTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, *normalTex, 0);

	unsigned int geometryAttachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, geometryAttachments);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// generate ssao fbo
	glGenFramebuffers(1, ssaoFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, *ssaoFBO);

	glGenTextures(1, ssaoColorTex);
	glBindTexture(GL_TEXTURE_2D, *ssaoColorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *ssaoColorTex, 0);

	unsigned int ssaoAttachments[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, ssaoAttachments);

	Graphics::checkFrambufferError();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	auto lerp = [](float a, float b, float t) { return a + t * (b - a); };

	//generate noise texture for use in ssao
	std::uniform_real_distribution<GLfloat> distribution(0.0, 1.0);
	std::default_random_engine generator;
	for (unsigned int j = 0; j < 64; ++j)
	{
		float x = distribution(generator) * 2.0f - 1.0f;
		float y = distribution(generator) * 2.0f - 1.0f;
		float z = distribution(generator);
		float radius = distribution(generator);

		glm::vec3 sample(x, y, z);
		sample = radius * glm::normalize(sample);
		float scale = float(j) / 64.0f;

		// scale samples s.t. they're more aligned to center of kernel
		scale = lerp(0.1f, 1.0f, scale * scale);
		sample *= scale;

		ssaoSamples[j] = sample;
	}

	glm::vec3 ssaoNoise[16];
	for (int j = 0; j < 16; j++) {
		// rotate around z-axis (in tangent space)
		glm::vec3 noise(distribution(generator) * 2.0f - 1.0f, distribution(generator) * 2.0f - 1.0f, 0.0f);
		ssaoNoise[j] = noise;
	}

	glGenTextures(1, ssaoNoiseTex);
	glBindTexture(GL_TEXTURE_2D, *ssaoNoiseTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glBindTexture(GL_TEXTURE_2D, 0);

	Graphics::checkError();

	*created = true;
}

void Graphics::destroy(Camera* camera,
					   GLuint* mainFBO,
					   GLuint* colorTex,
					   GLuint* depthTex,
					   GLuint* colorPickingFBO,
					   GLuint* colorPickingTex,
					   GLuint* colorPickingDepthTex,
					   GLuint* geometryFBO,
					   GLuint* positionTex,
					   GLuint* normalTex,
					   GLuint* ssaoFBO,
					   GLuint* ssaoColorTex,
					   GLuint* ssaoNoiseTex,
					   bool* created)
{
	// detach textures from their framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, *mainFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, *colorPickingFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, *geometryFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, *ssaoFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// delete frambuffers
	glDeleteFramebuffers(1, mainFBO);
	glDeleteFramebuffers(1, colorPickingFBO);
	glDeleteFramebuffers(1, geometryFBO);
	glDeleteFramebuffers(1, ssaoFBO);

	// delete textures
	glDeleteTextures(1, colorTex);
	glDeleteTextures(1, depthTex);
	glDeleteTextures(1, colorPickingTex);
	glDeleteTextures(1, colorPickingDepthTex);
	glDeleteTextures(1, positionTex);
	glDeleteTextures(1, normalTex);
	glDeleteTextures(1, ssaoColorTex);
	glDeleteTextures(1, ssaoNoiseTex);

	*created = false;
}

void Graphics::readColorPickingPixel(const Camera* camera, int x, int y, Color32* color)
{
	//glBindTexture(GL_TEXTURE_2D, camera->getNativeGraphicsColorPickingTex());


	glBindFramebuffer(GL_FRAMEBUFFER, camera->getNativeGraphicsColorPickingFBO());
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color);
	Graphics::checkError();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::create(Texture2D* texture, GLuint* tex, bool* created)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_2D, *tex);

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glBindTexture(GL_TEXTURE_2D, 0);

	*created = true;
}

void Graphics::destroy(Texture2D* texture, GLuint* tex, bool* created)
{
	glDeleteTextures(1, tex);

	*created = false;
}

void Graphics::readPixels(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glGetTextureImage(texture->getNativeGraphics(), 0, openglFormat, GL_UNSIGNED_BYTE, width*height*numChannels, &rawTextureData[0]);
	
	texture->setRawTextureData(rawTextureData, width, height, format);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::apply(Texture2D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glTexImage2D(GL_TEXTURE_2D, 0, openglFormat, width, height, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_2D, 0);
}

 void Graphics::create(Texture3D* texture, GLuint* tex, bool* created)
 {
 	int width = texture->getWidth();
 	int height = texture->getHeight();
 	int depth = texture->getDepth();
 	TextureFormat format = texture->getFormat();
 	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

 	glGenTextures(1, tex);
 	glBindTexture(GL_TEXTURE_3D, *tex);

 	GLenum openglFormat = Graphics::getTextureFormat(format);

 	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
 	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

 	glBindTexture(GL_TEXTURE_3D, 0);

	*created = true;
 }

 void Graphics::destroy(Texture3D* texture, GLuint* tex, bool* created)
 {
 	glDeleteTextures(1, tex);

	*created = false;
 }

void Graphics::readPixels(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	int numChannels = texture->getNumChannels();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_3D, texture->getNativeGraphics());

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glGetTextureImage(texture->getNativeGraphics(), 0, openglFormat, GL_UNSIGNED_BYTE, width*height*depth*numChannels, &rawTextureData[0]);

	texture->setRawTextureData(rawTextureData, width, height, depth, format);
	
	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::apply(Texture3D* texture)
{
	int width = texture->getWidth();
	int height = texture->getHeight();
	int depth = texture->getDepth();
	TextureFormat format = texture->getFormat();
	std::vector<unsigned char> rawTextureData = texture->getRawTextureData();

	glBindTexture(GL_TEXTURE_3D, texture->getNativeGraphics());

	GLenum openglFormat = Graphics::getTextureFormat(format);

	glTexImage3D(GL_TEXTURE_3D, 0, openglFormat, width, height, depth, 0, openglFormat, GL_UNSIGNED_BYTE, &rawTextureData[0]);

	glBindTexture(GL_TEXTURE_3D, 0);
}

void Graphics::create(Cubemap* cubemap, GLuint* tex, bool* created)
{
	int width = cubemap->getWidth();
	TextureFormat format = cubemap->getFormat();
	std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

 	glGenTextures(1, tex);
 	glBindTexture(GL_TEXTURE_CUBE_MAP, *tex);

 	GLenum openglFormat = Graphics::getTextureFormat(format);

 	for (unsigned int i = 0; i < 6; i++){
 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[0]);
 	}

 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
 	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	*created = true;
}

void Graphics::destroy(Cubemap* cubemap, GLuint* tex, bool* created)
{
	glDeleteTextures(1, tex);

	*created = false;
}

void Graphics::readPixels(Cubemap* cubemap)
{
	int width = cubemap->getWidth();
	int numChannels = cubemap->getNumChannels();
	TextureFormat format = cubemap->getFormat();
	std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

 	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap->getNativeGraphics());

 	GLenum openglFormat = Graphics::getTextureFormat(format);

 	for (unsigned int i = 0; i < 6; i++){
 		glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[i*width*width*numChannels]);
 	}

 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Graphics::apply(Cubemap* cubemap)
{
	int width = cubemap->getWidth();
	TextureFormat format = cubemap->getFormat();
	std::vector<unsigned char> rawCubemapData = cubemap->getRawCubemapData();

 	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap->getNativeGraphics());

 	GLenum openglFormat = Graphics::getTextureFormat(format);

 	for (unsigned int i = 0; i < 6; i++){
 		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, openglFormat, width, width, 0, openglFormat, GL_UNSIGNED_BYTE, &rawCubemapData[0]);
 	}

 	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Graphics::create(Mesh* mesh, GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2, bool* created)
{
	glGenVertexArrays(1, vao);
	glBindVertexArray(*vao);
	glGenBuffers(1, vbo0);
	glGenBuffers(1, vbo1);
	glGenBuffers(1, vbo2);

	glBindVertexArray(*vao);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo0);
	glBufferData(GL_ARRAY_BUFFER, mesh->getVertices().size() * sizeof(float), &(mesh->getVertices()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, *vbo1);
	glBufferData(GL_ARRAY_BUFFER, mesh->getNormals().size() * sizeof(float), &(mesh->getNormals()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindBuffer(GL_ARRAY_BUFFER, *vbo2);
	glBufferData(GL_ARRAY_BUFFER, mesh->getTexCoords().size() * sizeof(float), &(mesh->getTexCoords()[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);

	*created = true;
}

void Graphics::destroy(Mesh* mesh, GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2, bool* created)
{

}

void Graphics::apply(Mesh* mesh)
{

}

void Graphics::render(World* world, Material* material, int variant, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query) 
{
	if(material == NULL){
		std::cout << "Material is NULL" << std::endl;
		return;
	}

	Shader* shader = world->getAsset<Shader>(material->getShaderId());

	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->getId().toString() << " has not been compiled." << std::endl;
		return;
	}

	shader->use(variant);
	shader->setMat4("model", model);
	material->apply(world);
	/*shader->setFloat("material.shininess", material->shininess);
	shader->setVec3("material.ambient", material->ambient);
	shader->setVec3("material.ambient", material->diffuse);
	shader->setVec3("material.ambient", material->specular);

	Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
	if(mainTexture != NULL){
		shader->setInt("material.mainTexture", 0);

		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, mainTexture->handle.handle);
	}

	Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
	if(normalMap != NULL){
		shader->setInt("material.normalMap", 1);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, normalMap->handle.handle);
	}

	Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);
	if(specularMap != NULL){
		shader->setInt("material.specularMap", 2);

		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, specularMap->handle.handle);
	}*/

	/*if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}*/

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, numVertices);
	glBindVertexArray(0);

	//if(query != NULL){
	//	glEndQuery(GL_TIME_ELAPSED);

	//	GLint done = 0;
	//    while (!done) {
	//	    glGetQueryObjectiv(query->mQueryId, 
	//	            GL_QUERY_RESULT_AVAILABLE, 
	//	            &done);
	//	}

	//	// get the query result
	//	GLuint64 elapsedTime; // in nanoseconds
	//	glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

	//	query->mTotalElapsedTime += elapsedTime / 1000000.0f;
	//	query->mNumDrawCalls++;
	//	query->mVerts += numVertices;
	//	query->mTris += numVertices / 3;
	//}

	Graphics::checkError();
}

void Graphics::render(World* world, Shader* shader, int variant, Texture2D* texture, glm::mat4 model, GLuint vao, int numVertices, GraphicsQuery* query)
{
	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->getId().toString() << " has not been compiled." << std::endl;
		return;
	}

	shader->use(variant);
	shader->setMat4("model", model);

	if(texture != NULL){
		shader->setInt("texture0", 0);


		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, texture->getNativeGraphics());
	}

	/*if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}*/

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, numVertices);
	glBindVertexArray(0);

	//if(query != NULL){
	//	glEndQuery(GL_TIME_ELAPSED);

	//	GLint done = 0;
	//    while (!done) {
	//	    glGetQueryObjectiv(query->mQueryId, 
	//	            GL_QUERY_RESULT_AVAILABLE, 
	//	            &done);
	//	}

	//	// get the query result
	//	GLuint64 elapsedTime; // in nanoseconds
	//	glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

	//	query->mTotalElapsedTime += elapsedTime / 1000000.0f;
	//	query->mNumDrawCalls++;
	//	query->mVerts += numVertices;
	//	query->mTris += numVertices / 3;
	//}

	Graphics::checkError();
}

void Graphics::render(World* world, Shader* shader, int variant, glm::mat4 model, GLuint vao, GLenum mode, int numVertices, GraphicsQuery* query)
{
	if(shader == NULL){
		std::cout << "Shader is NULL" << std::endl;
		return;
	}

	if(!shader->isCompiled()){
		std::cout << "Shader " << shader->getId().toString() << " has not been compiled." << std::endl;
		return;
	}

	shader->use(variant);
	shader->setMat4("model", model);

	/*if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}*/

	glBindVertexArray(vao);
	glDrawArrays(mode, 0, numVertices);
	glBindVertexArray(0);

	//if(query != NULL){
	//	glEndQuery(GL_TIME_ELAPSED);

	//	GLint done = 0;
	//    while (!done) {
	//	    glGetQueryObjectiv(query->mQueryId,
	//	            GL_QUERY_RESULT_AVAILABLE, 
	//	            &done);
	//	}

	//	// get the query result
	//	GLuint64 elapsedTime; // in nanoseconds
	//	glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

	//	query->mTotalElapsedTime += elapsedTime / 1000000.0f;
	//	query->mNumDrawCalls++;
	//	query->mVerts += numVertices;
	//	if(mode == GL_TRIANGLES){
	//		query->mTris += numVertices / 3;
	//	}
	//	else if(mode == GL_LINES){
	//		query->mLines += numVertices / 2;
	//	}
	//	else if(mode == GL_POINTS){
	//		query->mPoints += numVertices;
	//	}
	//}

	Graphics::checkError();
}

void Graphics::renderText(World* world, Camera* camera, Font* font, std::string text, float x, float y, float scale, glm::vec3 color)
{
	if(!font->mShader.isCompiled()){
		std::cout << "Shader " << font->mShader.getId().toString() << " has not been compiled." << std::endl;
		return;
	}

	glm::mat4 ortho = glm::ortho(0.0f, (float)camera->mViewport.mWidth, 0.0f, (float)camera->mViewport.mHeight);

	font->mShader.use(ShaderVariant::None);
	font->mShader.setMat4("projection", ortho);
	font->mShader.setVec3("textColor", color);

	glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(font->mVao);

    // Iterate through all characters
    std::string::const_iterator it;
    for (it = text.begin(); it != text.end(); it++) 
    {
        Character ch = font->getCharacter(*it);//Characters[*it];

        GLfloat xpos = x + ch.mBearing.x * scale;
        GLfloat ypos = y - (ch.mSize.y - ch.mBearing.y) * scale;

        GLfloat w = ch.mSize.x * scale;
        GLfloat h = ch.mSize.y * scale;
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
        glBindTexture(GL_TEXTURE_2D, ch.mGlyphId);

        // Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, font->mVbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // Be sure to use glBufferSubData and not glBufferData

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.mAdvance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glBindVertexArray(0);

    Graphics::checkError();
}












// void Graphics::render(World* world, Material* material, LightType lightType, glm::mat4 model, int start, GLsizei size, GraphicsQuery* query)
// {
// 	if(material == NULL){
// 		std::cout << "Material is NULL" << std::endl;
// 		return;
// 	}

// 	Shader* shader = world->getAsset<Shader>(material->shaderId);

// 	if(shader == NULL){
// 		std::cout << "Shader is NULL" << std::endl;
// 		return;
// 	}

// 	if(!shader->isCompiled()){
// 		std::cout << "Shader " << shader->assetId.toString() << " has not been compiled." << std::endl;
// 		return;
// 	}

// 	Graphics::use(shader, lightType);
// 	Graphics::setMat4(shader, lightType, "model", model);
// 	Graphics::setFloat(shader, lightType, "material.shininess", material->shininess);
// 	Graphics::setVec3(shader, lightType, "material.ambient", material->ambient);
// 	Graphics::setVec3(shader, lightType, "material.diffuse", material->diffuse);
// 	Graphics::setVec3(shader, lightType, "material.specular", material->specular);

// 	Texture2D* mainTexture = world->getAsset<Texture2D>(material->textureId);
// 	if(mainTexture != NULL){
// 		Graphics::setInt(shader, lightType, "material.mainTexture", 0);

// 		glActiveTexture(GL_TEXTURE0 + 0);
// 		glBindTexture(GL_TEXTURE_2D, mainTexture->handle.handle);
// 	}

// 	Texture2D* normalMap = world->getAsset<Texture2D>(material->normalMapId);
// 	if(normalMap != NULL){

// 		Graphics::setInt(shader, lightType, "material.normalMap", 1);

// 		glActiveTexture(GL_TEXTURE0 + 1);
// 		glBindTexture(GL_TEXTURE_2D, normalMap->handle.handle);
// 	}

// 	Texture2D* specularMap = world->getAsset<Texture2D>(material->specularMapId);
// 	if(specularMap != NULL){

// 		Graphics::setInt(shader, lightType, "material.specularMap", 2);

// 		glActiveTexture(GL_TEXTURE0 + 2);
// 		glBindTexture(GL_TEXTURE_2D, specularMap->handle.handle);
// 	}

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = size / 3;
// 	GLint startIndex = start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }

// void Graphics::render(World* world, RenderObject renderObject, ShaderVariant variant, GraphicsQuery* query)
// {
// 	Material* material = world->getAssetByIndex<Material>(renderObject.materialIndex);

// 	GLuint shaderProgram = renderObject.shaders[(int)variant];

// 	Graphics::use(shaderProgram);
// 	Graphics::setMat4(shaderProgram, "model", renderObject.model);
// 	Graphics::setFloat(shaderProgram, "material.shininess", material->shininess);
// 	Graphics::setVec3(shaderProgram, "material.ambient", material->ambient);
// 	Graphics::setVec3(shaderProgram, "material.diffuse", material->diffuse);
// 	Graphics::setVec3(shaderProgram, "material.specular", material->specular);	

// 	if(renderObject.mainTexture != -1){
// 		Graphics::setInt(shaderProgram, "material.mainTexture", 0);

// 		glActiveTexture(GL_TEXTURE0 + 0);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.mainTexture);
// 	}

// 	if(renderObject.normalMap != -1){
// 		Graphics::setInt(shaderProgram, "material.normalMap", 1);

// 		glActiveTexture(GL_TEXTURE0 + 1);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.normalMap);		
// 	}

// 	if(renderObject.specularMap != -1){
// 		Graphics::setInt(shaderProgram, "material.specularMap", 2);

// 		glActiveTexture(GL_TEXTURE0 + 2);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.specularMap);		
// 	}

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = renderObject.size / 3;
// 	GLint startIndex = renderObject.start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }

// void Graphics::render(World* world, RenderObject renderObject, ShaderVariant variant, GLuint* shadowMaps, int shadowMapCount, GraphicsQuery* query)
// {
// 	Material* material = world->getAssetByIndex<Material>(renderObject.materialIndex);

// 	GLuint shaderProgram = renderObject.shaders[(int)variant];

// 	Graphics::use(shaderProgram);
// 	Graphics::setMat4(shaderProgram, "model", renderObject.model);
// 	Graphics::setFloat(shaderProgram, "material.shininess", material->shininess);
// 	Graphics::setVec3(shaderProgram, "material.ambient", material->ambient);
// 	Graphics::setVec3(shaderProgram, "material.diffuse", material->diffuse);
// 	Graphics::setVec3(shaderProgram, "material.specular", material->specular);	

// 	if(renderObject.mainTexture != -1){
// 		Graphics::setInt(shaderProgram, "material.mainTexture", 0);

// 		glActiveTexture(GL_TEXTURE0 + 0);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.mainTexture);
// 	}

// 	if(renderObject.normalMap != -1){
// 		Graphics::setInt(shaderProgram, "material.normalMap", 1);

// 		glActiveTexture(GL_TEXTURE0 + 1);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.normalMap);		
// 	}

// 	if(renderObject.specularMap != -1){
// 		Graphics::setInt(shaderProgram, "material.specularMap", 2);

// 		glActiveTexture(GL_TEXTURE0 + 2);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.specularMap);		
// 	}

// 	for(int i = 0; i < shadowMapCount; i++){
// 		//std::cout << "shadowMap[" + std::to_string(i) + "]: " << 3 + i << std::endl;
// 		Graphics::setInt(shaderProgram, "shadowMap[" + std::to_string(i) + "]", 3 + i);

// 		glActiveTexture(GL_TEXTURE0 + 3 + i);
// 		glBindTexture(GL_TEXTURE_2D, shadowMaps[i]);
// 	}

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = renderObject.size / 3;
// 	GLint startIndex = renderObject.start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }
// void Graphics::render(World* world, RenderObject renderObject, GLuint shaderProgram, GraphicsQuery* query)
// {
// 	Material* material = world->getAssetByIndex<Material>(renderObject.materialIndex);

// 	Graphics::setMat4(shaderProgram, "model", renderObject.model);
// 	Graphics::setFloat(shaderProgram, "material.shininess", material->shininess);
// 	Graphics::setVec3(shaderProgram, "material.ambient", material->ambient);
// 	Graphics::setVec3(shaderProgram, "material.diffuse", material->diffuse);
// 	Graphics::setVec3(shaderProgram, "material.specular", material->specular);	

// 	if(renderObject.mainTexture != -1){
// 		Graphics::setInt(shaderProgram, "material.mainTexture", 0);

// 		glActiveTexture(GL_TEXTURE0 + 0);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.mainTexture);
// 	}

// 	if(renderObject.normalMap != -1){
// 		Graphics::setInt(shaderProgram, "material.normalMap", 1);

// 		glActiveTexture(GL_TEXTURE0 + 1);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.normalMap);		
// 	}

// 	if(renderObject.specularMap != -1){
// 		Graphics::setInt(shaderProgram, "material.specularMap", 2);

// 		glActiveTexture(GL_TEXTURE0 + 2);
// 		glBindTexture(GL_TEXTURE_2D, (GLuint)renderObject.specularMap);		
// 	}

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = renderObject.size / 3;
// 	GLint startIndex = renderObject.start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }


// void Graphics::render(World* world, Shader* shader, ShaderVariant variant, glm::mat4 model, int start, GLsizei size, GraphicsQuery* query)
// {
// 	Graphics::use(shader, variant);
// 	Graphics::setMat4(shader, variant, "model", model);

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = size / 3;
// 	GLint startIndex = start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }

// void Graphics::render(World* world, Shader* shader, ShaderVariant variant, glm::mat4 model, glm::mat4 view, glm::mat4 projection, int start, GLsizei size, GraphicsQuery* query)
// {
// 	Graphics::use(shader, variant);
// 	Graphics::setMat4(shader, variant, "model", model);
// 	Graphics::setMat4(shader, variant, "view", view);
// 	Graphics::setMat4(shader, variant, "projection", projection);

// 	if(world->debug && query != NULL){
// 		glBeginQuery(GL_TIME_ELAPSED, query->queryId);
// 	}

// 	GLsizei numVertices = size / 3;
// 	GLint startIndex = start / 3;

// 	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);

// 	if(world->debug && query != NULL){
// 		glEndQuery(GL_TIME_ELAPSED);

// 		GLint done = 0;
// 	    while (!done) {
// 		    glGetQueryObjectiv(query->queryId, 
// 		            GL_QUERY_RESULT_AVAILABLE, 
// 		            &done);
// 		}

// 		// get the query result
// 		GLuint64 elapsedTime; // in nanoseconds
// 		glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

// 		query->totalElapsedTime += elapsedTime / 1000000.0f;
// 		query->numDrawCalls++;
// 		query->verts += numVertices;
// 		query->tris += numVertices / 3;
// 	}

// 	Graphics::checkError();
// }

void Graphics::render(World* world, RenderObject renderObject, GraphicsQuery* query)
{
	//if(world->debug && query != NULL){
	//	glBeginQuery(GL_TIME_ELAPSED, query->queryId);
	//}

	GLsizei numVertices = renderObject.size / 3;
	GLint startIndex = renderObject.start / 3;

	glBindVertexArray(renderObject.vao);
	glDrawArrays(GL_TRIANGLES, startIndex, numVertices);
	glBindVertexArray(0);

	query->mNumDrawCalls++;
	query->mVerts += numVertices;
	query->mTris += numVertices / 3;

	//if(world->debug && query != NULL){
	//	glEndQuery(GL_TIME_ELAPSED);

	//	GLint done = 0;
	//    while (!done) {
	//	    glGetQueryObjectiv(query->queryId, 
	//	            GL_QUERY_RESULT_AVAILABLE, 
	//	            &done);
	//	}

	//	// get the query result
	//	GLuint64 elapsedTime; // in nanoseconds
	//	glGetQueryObjectui64v(query->queryId, GL_QUERY_RESULT, &elapsedTime);

	//	query->totalElapsedTime += elapsedTime / 1000000.0f;
	//	query->numDrawCalls++;
	//	query->verts += numVertices;
	//	query->tris += numVertices / 3;
	//}

	Graphics::checkError();
}
























void Graphics::draw(World* world, MeshRenderer* meshRenderer, GLuint fbo)
{
	/*if (world == NULL || meshRenderer == NULL) {
		return;
	}

	Transform* transform = meshRenderer->getComponent<Transform>();
	Material* material = world->getAsset<Material>(meshRenderer->getMaterial());
	Mesh* mesh = world->getAsset<Mesh>(meshRenderer->getMesh());

	if (transform == NULL || material == NULL || mesh == NULL) {
		return;
	}

	Shader* shader = world->getAsset<Shader>(material->getShaderId());

	if (shader == NULL) {
		return;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	int shaderProgram = shader->getProgramFromVariant(0);

	shader->use(shaderProgram);
	shader->setMat4("model", transform->getModelMatrix());

	material->apply(world);

	int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(0);
	int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(0);

	GLsizei numVertices = (subMeshVertexEndIndex - subMeshVertexStartIndex) / 3;

	glBindVertexArray(mesh->getNativeGraphicsVAO());
	glDrawArrays(GL_TRIANGLES, 0, numVertices);
	glBindVertexArray(0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);*/
}