#ifndef DIRECTX_FRAMEBUFFER_H__
#define DIRECTX_FRAMEBUFFER_H__

#include "../../Framebuffer.h"

namespace PhysicsEngine
{
	class DirectXFramebuffer : public Framebuffer
	{
	public:
		DirectXFramebuffer(int width, int height);
		DirectXFramebuffer(int width, int height, int numColorTex, bool addDepthTex);
		~DirectXFramebuffer();

		void clearColor(Color color) override;
        void clearColor(float r, float g, float b, float a) override;
        void clearDepth(float depth) override;
		void bind() override;
		void unbind() override;
        void setViewport(int x, int y, int width, int height) override;

		TextureHandle *getColorTex(size_t i = 0) override;
        TextureHandle *getDepthTex() override;
		void* getHandle() override;
	};
}

#endif