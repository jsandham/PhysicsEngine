//#include <iostream>
//#include "CubeMapFramebuffer.h"
//
//using namespace PhysicsEngine;
//
//CubeMapFramebuffer::CubeMapFramebuffer(int width) : Framebuffer(width, width)
//{
//	init();
//}
//
//CubeMapFramebuffer::~CubeMapFramebuffer()
//{
//	delete texture;
//
//	glDeleteFramebuffers(1, &handle);
//}
//
//void CubeMapFramebuffer::init()
//{
//	texture = new CubeMapTexture(width);
//
//	glGenFramebuffers(1, &handle);
//	glBindFramebuffer(GL_FRAMEBUFFER, handle);
//	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture->getHandle(), 0);
//	
//	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
//		std::cout << "ERROR: CUBE MAP FRAMEBUFFER IS NOT COMPLETE " << framebufferStatus << std::endl;
//	}
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//}
//
//void CubeMapFramebuffer::bind()
//{
//	glViewport(0, 0, width, height);
//	glBindFramebuffer(GL_FRAMEBUFFER, handle);
//}
//
//void CubeMapFramebuffer::unbind()
//{
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//}
//
//void CubeMapFramebuffer::clear()
//{
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//}
//
//CubeMapTexture* CubeMapFramebuffer::getTexture()
//{
//	return texture;
//}