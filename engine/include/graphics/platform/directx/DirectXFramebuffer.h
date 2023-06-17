#ifndef DIRECTX_FRAMEBUFFER_H__
#define DIRECTX_FRAMEBUFFER_H__

#include "../../Framebuffer.h"

#include <vector>
#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
	class DirectXFramebuffer : public Framebuffer
	{
      private:
        std::vector<ID3D11RenderTargetView*> mRenderTargetViews;
        std::vector<ID3D11RenderTargetView*> mNullRenderTargetViews;
        ID3D11DepthStencilView* mDepthStencilView;

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
	};
}

#endif