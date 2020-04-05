#include <iostream>
#include <algorithm>
#include <GL/glew.h>

#include "../../include/core/Log.h"
#include "../../include/core/InternalShaders.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/graphics/GraphicsState.h"

using namespace PhysicsEngine;


void DebugWindow::init()
{
	mShader.setVertexShader(InternalShaders::windowVertexShader);
	mShader.setFragmentShader(InternalShaders::windowFragmentShader);
	mShader.compile();

	mX = fmin(fmax(mX, 0.0f), 1.0f);
	mY = fmin(fmax(mY, 0.0f), 1.0f);
	mWidth = fmin(fmax(mWidth, 0.0f), 1.0f);
	mHeight = fmin(fmax(mHeight, 0.0f), 1.0f);

	float x_ndc = 2.0f * mX - 1.0f; 
	float y_ndc = 1.0f - 2.0f * mY; 

	float width_ndc = 2.0f * mWidth;
	float height_ndc = 2.0f * mHeight;

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

	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);

	glGenBuffers(1, &mVertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexVBO);
	glBufferData(GL_ARRAY_BUFFER, 18*sizeof(float), &(vertices[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glGenBuffers(1, &mTexCoordVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mTexCoordVBO);
	glBufferData(GL_ARRAY_BUFFER, 12*sizeof(float), &(texCoords[0]), GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);
}

void PerformanceGraph::init()
{
	mShader.setVertexShader(InternalShaders::graphVertexShader);
	mShader.setFragmentShader(InternalShaders::graphFragmentShader);
	mShader.compile();

	mX = fmin(fmax(mX, 0.0f), 1.0f);
	mY = fmin(fmax(mY, 0.0f), 1.0f);
	mWidth = fmin(fmax(mWidth, 0.0f), 1.0f);
	mHeight = fmin(fmax(mHeight, 0.0f), 1.0f);
	mRangeMin = fmin(mRangeMin, mRangeMax);
	mRangeMax = fmax(mRangeMin, mRangeMax);
	mCurrentSample = 0.0f;
	mNumberOfSamples = std::max(2, mNumberOfSamples);

	mSamples.resize(18*mNumberOfSamples - 18);

	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);

	glGenBuffers(1, &mVBO);
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	glBufferData(GL_ARRAY_BUFFER, mSamples.size()*sizeof(float), &(mSamples[0]), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

	glBindVertexArray(0);
}

void PerformanceGraph::add(float sample)
{
	float oldSample = mCurrentSample;
	mCurrentSample = fmin(fmax(sample, mRangeMin), mRangeMax);

	float dx = mWidth / (mNumberOfSamples - 1);
	for(int i = 0; i < mNumberOfSamples - 2; i++){
		mSamples[18*i] = mSamples[18*(i+1)] - dx;
		mSamples[18*i + 1] = mSamples[18*(i+1) + 1];
		mSamples[18*i + 2] = mSamples[18*(i+1) + 2];

		mSamples[18*i + 3] = mSamples[18*(i+1) + 3] - dx;
		mSamples[18*i + 4] = mSamples[18*(i+1) + 4];
		mSamples[18*i + 5] = mSamples[18*(i+1) + 5];

		mSamples[18*i + 6] = mSamples[18*(i+1) + 6] - dx;
		mSamples[18*i + 7] = mSamples[18*(i+1) + 7];
		mSamples[18*i + 8] = mSamples[18*(i+1) + 8];

		mSamples[18*i + 9] = mSamples[18*(i+1) + 9] - dx;
		mSamples[18*i + 10] = mSamples[18*(i+1) + 10];
		mSamples[18*i + 11] = mSamples[18*(i+1) + 11];

		mSamples[18*i + 12] = mSamples[18*(i+1) + 12] - dx;
		mSamples[18*i + 13] = mSamples[18*(i+1) + 13];
		mSamples[18*i + 14] = mSamples[18*(i+1) + 14];

		mSamples[18*i + 15] = mSamples[18*(i+1) + 15] - dx;
		mSamples[18*i + 16] = mSamples[18*(i+1) + 16];
		mSamples[18*i + 17] = mSamples[18*(i+1) + 17];
	}

	float dz1 = 1.0f - (mCurrentSample - mRangeMin) / (mRangeMax - mRangeMin);
	float dz2 = 1.0f - (oldSample - mRangeMin) / (mRangeMax - mRangeMin);

	float x_ndc = 2.0f * mX - 1.0f;
	float y0_ndc = 1.0f - 2.0f * (mY + mHeight);
	float y1_ndc = 1.0f - 2.0f * (mY + mHeight * dz1);
	float y2_ndc = 1.0f - 2.0f * (mY + mHeight * dz2);

	mSamples[18*(mNumberOfSamples - 2)] = x_ndc + dx * (mNumberOfSamples - 2);
	mSamples[18*(mNumberOfSamples - 2) + 1] = y2_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 2] = 0.0f;

	mSamples[18*(mNumberOfSamples - 2) + 3] = x_ndc + dx * (mNumberOfSamples - 2);
	mSamples[18*(mNumberOfSamples - 2) + 4] = y0_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 5] = 0.0f;

	mSamples[18*(mNumberOfSamples - 2) + 6] = x_ndc + dx * (mNumberOfSamples - 1);
	mSamples[18*(mNumberOfSamples - 2) + 7] = y0_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 8] = 0.0f;

	mSamples[18*(mNumberOfSamples - 2) + 9] = x_ndc + dx * (mNumberOfSamples - 2);
	mSamples[18*(mNumberOfSamples - 2) + 10] = y2_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 11] = 0.0f;

	mSamples[18*(mNumberOfSamples - 2) + 12] = x_ndc + dx * (mNumberOfSamples - 1);
	mSamples[18*(mNumberOfSamples - 2) + 13] = y0_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 14] = 0.0f;

	mSamples[18*(mNumberOfSamples - 2) + 15] = x_ndc + dx * (mNumberOfSamples - 1);
	mSamples[18*(mNumberOfSamples - 2) + 16] = y1_ndc;
	mSamples[18*(mNumberOfSamples - 2) + 17] = 0.0f;

	glBindVertexArray(mVAO);
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);

	glBufferSubData(GL_ARRAY_BUFFER, 0, mSamples.size()*sizeof(float), &(mSamples[0]));

	glBindVertexArray(0);
}

LineBuffer::LineBuffer()
{
	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);
	glGenBuffers(1, &mVBO);
	glBindVertexArray(0);
}

LineBuffer::~LineBuffer()
{
	glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mVBO);
}

MeshBuffer::MeshBuffer()
{
	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);
	glGenBuffers(3, &mVBO[0]);
	glBindVertexArray(0);
}

MeshBuffer::~MeshBuffer()
{
	glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(3, &mVBO[0]);
}

int MeshBuffer::getStartIndex(Guid meshId)
{
	for(int i = 0; i < mMeshIds.size(); i++){
		if(mMeshIds[i] == meshId){
			return mStart[i];
		}
	}

	return -1;
}

Sphere MeshBuffer::getBoundingSphere(Guid meshId)
{
	for(int i = 0; i < mMeshIds.size(); i++){
		if(mMeshIds[i] == meshId){
			return mBoundingSpheres[i];
		}
	}

	Sphere sphere;

	return sphere;
}

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

	if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, numVertices);
	glBindVertexArray(0);

	if(query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->mQueryId, 
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime; // in nanoseconds
		glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

		query->mTotalElapsedTime += elapsedTime / 1000000.0f;
		query->mNumDrawCalls++;
		query->mVerts += numVertices;
		query->mTris += numVertices / 3;
	}

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

	if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, numVertices);
	glBindVertexArray(0);

	if(query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->mQueryId, 
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime; // in nanoseconds
		glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

		query->mTotalElapsedTime += elapsedTime / 1000000.0f;
		query->mNumDrawCalls++;
		query->mVerts += numVertices;
		query->mTris += numVertices / 3;
	}

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

	if(query != NULL){
		glBeginQuery(GL_TIME_ELAPSED, query->mQueryId);
	}

	glBindVertexArray(vao);
	glDrawArrays(mode, 0, numVertices);
	glBindVertexArray(0);

	if(query != NULL){
		glEndQuery(GL_TIME_ELAPSED);

		GLint done = 0;
	    while (!done) {
		    glGetQueryObjectiv(query->mQueryId,
		            GL_QUERY_RESULT_AVAILABLE, 
		            &done);
		}

		// get the query result
		GLuint64 elapsedTime; // in nanoseconds
		glGetQueryObjectui64v(query->mQueryId, GL_QUERY_RESULT, &elapsedTime);

		query->mTotalElapsedTime += elapsedTime / 1000000.0f;
		query->mNumDrawCalls++;
		query->mVerts += numVertices;
		if(mode == GL_TRIANGLES){
			query->mTris += numVertices / 3;
		}
		else if(mode == GL_LINES){
			query->mLines += numVertices / 2;
		}
		else if(mode == GL_POINTS){
			query->mPoints += numVertices;
		}
	}

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
    glBindVertexArray(font->mVao.handle);

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
        glBindTexture(GL_TEXTURE_2D, ch.mGlyphId.handle);

        // Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, font->mVbo.handle);
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

	//Graphics::checkError();
}