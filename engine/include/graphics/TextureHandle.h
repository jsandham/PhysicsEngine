#ifndef TEXTURE_HANDLE_H__
#define TEXTURE_HANDLE_H__

#include <vector>

#include "../core/Texture.h"

namespace PhysicsEngine
{
	class TextureHandle
	{
	protected:
        TextureFormat mFormat;
		TextureWrapMode mWrapMode;
		TextureFilterMode mFilterMode;
		int mAnisoLevel;
		int mWidth;
		int mHeight;

	public:
		TextureHandle();
		TextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                    TextureFilterMode filterMode);
        TextureHandle(const TextureHandle &other) = delete;
        TextureHandle &operator=(const TextureHandle &other) = delete;
		virtual ~TextureHandle() = 0;

		TextureFormat getFormat() const;
        TextureWrapMode getWrapMode() const;
        TextureFilterMode getFilterMode() const;
        int getAnisoLevel() const;
        int getWidth() const;
        int getHeight() const;

		virtual void load(TextureFormat format,
			TextureWrapMode wrapMode,
			TextureFilterMode filterMode,
			int width,
			int height,
			const std::vector<unsigned char>& data) = 0;
        virtual void update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel) = 0;
		virtual void readPixels(std::vector<unsigned char>& data) = 0;
		virtual void writePixels(const std::vector<unsigned char>& data) = 0;
		virtual void bind(unsigned int texUnit) = 0;
		virtual void unbind(unsigned int texUnit) = 0;

		virtual void* getHandle() = 0;

		static TextureHandle* create();
        static TextureHandle* create(int width, int height, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode);
	};
}

#endif