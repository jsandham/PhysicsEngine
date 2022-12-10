#ifndef RENDER_CONTEXT_DIRECTX_H__
#define RENDER_CONTEXT_DIRECTX_H__

#include "../../RenderContext.h"

#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
	class DirectXRenderContext : public RenderContext
	{
	private:
		ID3D11Device* mD3DDevice;
		ID3D11DeviceContext* mD3DDeviceContext;
		ID3D11RenderTargetView* mD3DTarget;

		IDXGIDevice* mDevice;
		IDXGIAdapter* mAdapter;
		IDXGISwapChain* mSwapChain;

		unsigned int mSwapInterval;

	public:
		DirectXRenderContext(void* window);
		~DirectXRenderContext();

		void present();
		void turnVsyncOn();
		void turnVsyncOff();

		ID3D11Device* getD3DDevice() { return mD3DDevice; }
		ID3D11DeviceContext* getD3DDeviceContext() { return mD3DDeviceContext; }

		IDXGIDevice* getDevice() { return mDevice; }
		IDXGIAdapter* getAdapter() { return mAdapter; }

		static DirectXRenderContext* get() { return (DirectXRenderContext*)sContext; }
	};
}

#endif // RENDER_CONTEXT_DIRECTX_H__