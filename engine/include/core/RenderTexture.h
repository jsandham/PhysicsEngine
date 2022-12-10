#ifndef RENDER_TEXTURE_H__
#define RENDER_TEXTURE_H__

#include "Texture.h"
#include "../graphics/Framebuffer.h"

namespace PhysicsEngine
{
    struct RenderTextureTargets
    {
        Framebuffer *mMainFBO;
        //unsigned int mMainFBO;
        //unsigned int mColorTex;
        //unsigned int mDepthTex;
    };

    class RenderTexture : public Texture
    {
    private:
        RenderTextureTargets mTargets;
        int mWidth;
        int mHeight;

    public:
        RenderTexture(World *world, const Id &id);
        RenderTexture(World *world, const Guid &guid, const Id &id);
        RenderTexture(World *world, const Id &id, int width, int height);
        RenderTexture(World *world, const Id &id, int width, int height, TextureFormat format);
        ~RenderTexture();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void writeToPNG(const std::string& filepath) const;
        void writeToJPG(const std::string& filepath) const;
        void writeToBMP(const std::string& filepath) const;

        int getWidth() const;
        int getHeight() const;

        void create() override;
        void update() override;
        void readPixels() override;
        void writePixels() override;

        Framebuffer* getNativeGraphicsMainFBO() const;
        TextureHandle* getNativeGraphicsColorTex() const;
        TextureHandle* getNativeGraphicsDepthTex() const;
	};

    template <> struct AssetType<RenderTexture>
    {
        static constexpr int type = PhysicsEngine::RENDER_TEXTURE_TYPE;
    };

    template <> struct IsAssetInternal<RenderTexture>
    {
        static constexpr bool value = true;
    };
}

#endif