// #include <iostream>
// #include "../../include/graphics/Framebuffer.h"

// #include "../../include/core/Log.h"

// using namespace PhysicsEngine;

// Framebuffer::Framebuffer()
// {

// }

// Framebuffer::~Framebuffer()
// {

// }


// Framebuffer::Framebuffer(int width, int height)
// {
// 	this->width = width;
// 	this->height = height;
// }

// Framebuffer::~Framebuffer()
// {

// }

// void Framebuffer::generate()
// {
// 	glGenFramebuffers(1, &handle);
// }

// void Framebuffer::destroy()
// {
// 	glDeleteFramebuffers(1, &handle);
// }

// void Framebuffer::addAttachment(GLuint textureHandle, GLenum attachmentType, GLint level)
// {
// 	glFramebufferTexture(GL_FRAMEBUFFER, attachmentType, textureHandle, level);

// 	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
// 		Log::Error("Framebuffer: Depth cubemap framebuffer is not complete %d", framebufferStatus);
// 	}
// }

// void Framebuffer::addAttachment2D(GLuint textureHandle, GLenum attachmentType, GLenum textureTarget, GLint level)
// {
// 	glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentType, textureTarget, textureHandle, level);

// 	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
// 		Log::Error("Framebuffer: Depth cubemap framebuffer is not complete %d", framebufferStatus);
// 	}
// }

// void Framebuffer::bind()
// {
// 	glViewport(0, 0, width, height);
// 	glBindFramebuffer(GL_FRAMEBUFFER, handle);
// }

// void Framebuffer::unbind()
// {
// 	glBindFramebuffer(GL_FRAMEBUFFER, 0);
// }

// void Framebuffer::clearColorBuffer(glm::vec4 value)
// {
// 	glClearColor(value.x, value.y, value.z, value.w);
// 	glClear(GL_COLOR_BUFFER_BIT);
// }

// void Framebuffer::clearDepthBuffer(float value)
// {
// 	glClearDepth(value);
// 	glClear(GL_DEPTH_BUFFER_BIT);
// }

// int Framebuffer::getWidth() const
// {
// 	return width;
// }

// int Framebuffer::getHeight() const
// {
// 	return height;
// }