#ifndef DIRECTX_INDEX_BUFFER_H__
#define DIRECTX_INDEX_BUFFER_H__

#include "../../IndexBuffer.h"

#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
	class DirectXIndexBuffer : public IndexBuffer
	{
	private:
		D3D11_BUFFER_DESC mBufferDesc;
		D3D11_MAPPED_SUBRESOURCE mMappedSubresource;
		ID3D11Buffer* mBufferHandle;
		ID3D11InputLayout* mInputLayout;

	public:
		DirectXIndexBuffer();
		~DirectXIndexBuffer();

		void resize(size_t size) override;
		void setData(void* data, size_t offset, size_t size) override;
		void bind() override;
		void unbind() override;
		void* getBuffer() override;
	};
}

#endif