#ifndef RENDERER_API_WIN32_DIRECTX_H__
#define RENDERER_API_WIN32_DIRECTX_H__

#include "../RendererAPI.h"

#include <windows.h>
#include <d3d9.h>

namespace PhysicsEngine
{
	class RendererAPI_win32_directx : public RendererAPI
	{
	private:
		IDirect3D9* D3D;
		IDirect3DDevice9* Device;

	public:
		RendererAPI_win32_directx();
		~RendererAPI_win32_directx();

		void init(void* window) override;
		void update() override;
		void cleanup() override;
        void turnVsyncOn() override;
        void turnVsyncOff() override;
	};
}

#endif RENDERER_API_WIN32_DIRECTX_H__