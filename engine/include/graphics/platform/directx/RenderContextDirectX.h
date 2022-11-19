#ifndef RENDER_CONTEXT_DIRECTX_H__
#define RENDER_CONTEXT_DIRECTX_H__

#include "../../RenderContext.h"

#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
	class RenderContextDirectX : public RenderContext
	{
	private:
		ID3D11Device* mDevice;
		ID3D11DeviceContext* mDeviceContext;
		IDXGISwapChain* mSwapChain;
		ID3D11RenderTargetView* mTarget;

	public:
		RenderContextDirectX(void* window);
		~RenderContextDirectX();

		void present();
		void turnVsyncOn();
		void turnVsyncOff();

		static RenderContextDirectX* get() { return (RenderContextDirectX*)sContext; }
	};
}

#endif // RENDER_CONTEXT_DIRECTX_H__




//#ifndef RENDER_CONTEXT_WIN32_DIRECTX_H__
//#define RENDER_CONTEXT_WIN32_DIRECTX_H__
//
//#include "../RenderContext.h"
//
//#include <windows.h>
//#include <d3d9.h>
//
//namespace PhysicsEngine
//{
//	class RenderContext_win32_directx : public RenderContext
//	{
//	private:
//		IDirect3D9* D3D;
//		IDirect3DDevice9* Device;
//
//	public:
//        RenderContext_win32_directx();
//       ~RenderContext_win32_directx();
//
//		void init(void* window) override;
//		void update() override;
//		void cleanup() override;
//        void turnVsyncOn() override;
//        void turnVsyncOff() override;
//	};
//}
//
//#endif // RENDER_CONTEXT_WIN32_DIRECTX_H__