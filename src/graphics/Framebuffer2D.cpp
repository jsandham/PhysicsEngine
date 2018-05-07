//#include <iostream>
//#include "Framebuffer2D.h"
//
//using namespace PhysicsEngine;
//
//Framebuffer2D::Framebuffer2D(int width, int height) : Framebuffer(width, height)
//{
//	
//}
//
//Framebuffer2D::~Framebuffer2D()
//{
//}
//
//void Framebuffer2D::generate(RenderTexture* texture)
//{
//	glGenFramebuffers(1, &handle);
//	glBindFramebuffer(GL_FRAMEBUFFER, handle);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture->getHandle(), 0);
//
//	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
//		std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << framebufferStatus << std::endl;
//	}
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//}
//
//void Framebuffer2D::destroy()
//{
//	glDeleteFramebuffers(1, &handle);
//}
//
//void Framebuffer2D::bind()
//{
//	glViewport(0, 0, width, height);
//	glBindFramebuffer(GL_FRAMEBUFFER, handle);
//}
//
//void Framebuffer2D::unbind()
//{
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//}
//
//void Framebuffer2D::clear()
//{
//	glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//}
//
//void Framebuffer2D::setClearColor(glm::vec4 &color)
//{
//	clearColor = color;
//}
//


//void Framebuffer2D::generate(Texture2D* texture)
//{
//	texture = new Texture2D(width, height, 3);
//
//	texture->generate();
//
//	glGenFramebuffers(1, &handle);
//	glBindFramebuffer(GL_FRAMEBUFFER, handle);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture->getHandle(), 0);
//
//	if ((framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE){
//		std::cout << "ERROR: FRAMEBUFFER 2D IS NOT COMPLETE " << framebufferStatus << std::endl;
//	}
//	glBindFramebuffer(GL_FRAMEBUFFER, 0);
//}
