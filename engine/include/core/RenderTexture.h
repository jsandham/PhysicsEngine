//#ifndef __RENDERTEXTURE_H__
//#define __RENDERTEXTURE_H__
//
//#include "Texture.h"
//#include "Texture2D.h"
//#include "Framebuffer.h"
//
//namespace PhysicsEngine
//{
//	/*typedef enum RenderTextureFormat
//	{
//		DepthOnly,
//		RGB,
//		RGBA
//	};*/
//
//	template<class T>
//	class RenderTexture : public Texture
//	{
//		private:
//			T* colourTexture;
//			T* depthTexture;
//			Framebuffer* framebuffer;
//
//			int depth;
//			TextureFormat colorFormat;
//			
//		public:
//			RenderTexture(int width, int height, int depth)
//			{
//				this->width = width;
//				this->height = height;
//				this->depth = depth;
//				this->colorFormat = TextureFormat::RGB;
//
//				colourTexture = new T(width, height, colorFormat);
//				depthTexture = NULL;
//
//				if (depth > 0){
//					depthTexture = new T(width, height, TextureFormat::Depth);
//				}
//
//				framebuffer = new Framebuffer(width, height);
//			}
//
//			RenderTexture(int width, int height, int depth, TextureFormat colorFormat)
//			{
//				this->width = width;
//				this->height = height;
//				this->colorFormat = colorFormat;
//
//				colourTexture = NULL;
//				depthTexture = NULL;
//
//				if (colorFormat != TextureFormat::Depth){
//					colourTexture = new T(width, height, colorFormat);
//				}
//
//				if (depth > 0){
//					depthTexture = new T(width, height, TextureFormat::Depth);
//				}
//
//				framebuffer = new Framebuffer(width, height);
//			}
//
//			~RenderTexture()
//			{
//				if (colourTexture != NULL){
//					delete colourTexture;
//				}
//
//				if (depthTexture != NULL){
//					delete depthTexture;
//				}
//
//				delete framebuffer;
//			}
//
//			void generate()
//			{
//				framebuffer->generate();
//
//				framebuffer->bind();
//			
//				if (colourTexture != NULL){
//					colourTexture->generate();
//
//					if (colourTexture->getDimension() == TextureDimension::Tex2D){
//						framebuffer->addAttachment2D(colourTexture->getHandle(), GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0);
//					}
//					else if (colourTexture->getDimension() == TextureDimension::Cube){
//						framebuffer->addAttachment(colourTexture->getHandle(), GL_COLOR_ATTACHMENT0, 0);
//					}
//				}
//				else{
//					// do I need this if the framebuffer has no color attachment
//					glDrawBuffer(GL_NONE);
//					glReadBuffer(GL_NONE);
//				}
//			
//				if (depthTexture != NULL){
//					depthTexture->generate();
//
//					std::cout << "depth texture handle: " << depthTexture->getHandle() << std::endl;
//
//					if (depthTexture->getDimension() == TextureDimension::Tex2D){
//						framebuffer->addAttachment2D(depthTexture->getHandle(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0);
//					}
//					else if (depthTexture->getDimension() == TextureDimension::Cube){
//						framebuffer->addAttachment(depthTexture->getHandle(), GL_DEPTH_ATTACHMENT, 0);
//					}
//				}
//
//				framebuffer->unbind();
//			}
//			
//			void destroy()
//			{
//				framebuffer->destroy();
//			}
//			
//			void bind()
//			{
//				framebuffer->bind();
//			}
//			
//			void unbind()
//			{
//				framebuffer->unbind();
//			}
//
//			void clearColorBuffer(glm::vec4 value)
//			{
//				framebuffer->clearColorBuffer(value);
//			}
//
//			void clearDepthBuffer(float value)
//			{
//				framebuffer->clearDepthBuffer(value);
//			}
//
//			T* getColourTexture()
//			{
//				return colourTexture;
//			}
//
//			T* getDepthTexture()
//			{
//				return depthTexture;
//			}
//	};
//}
//
//#endif