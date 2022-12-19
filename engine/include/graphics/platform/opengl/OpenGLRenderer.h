#ifndef OPENGL_RENDERER_API_H__
#define OPENGL_RENDERER_API_H__

#include "../../Renderer.h"
#include "OpenGLRenderContext.h"
#include "OpenGLTextureHandle.h"
#include "OpenGLVertexBuffer.h"

namespace PhysicsEngine
{
    class OpenGLRenderer : public Renderer
	{
    private:
        OpenGLRenderContext* mContext;

      protected:
        void init_impl() override;
        void present_impl() override;
        void turnVsyncOn_impl() override;
        void turnVsyncOff_impl() override;
        void bindFramebuffer_impl(Framebuffer* fbo) override;
        void unbindFramebuffer_impl() override;
        void clearFrambufferColor_impl(const Color &color) override;
        void clearFrambufferColor_impl(float r, float g, float b, float a) override;
        void clearFramebufferDepth_impl(float depth) override;
        void setViewport_impl(int x, int y, int width, int height) override;
        
        void turnOn_impl(Capability capability) override;
        void turnOff_impl(Capability capability) override;
        void setBlending_impl(BlendingFactor source, BlendingFactor dest) override;
        void beginQuery_impl(unsigned int queryId) override;
        void endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime) override;
        void createGlobalCameraUniforms_impl(CameraUniform &uniform) override;
        void createGlobalLightUniforms_impl(LightUniform &uniform) override;
        void setGlobalCameraUniforms_impl(const CameraUniform &uniform) override;
        void setGlobalLightUniforms_impl(const LightUniform &uniform) override;
        void createScreenQuad_impl(unsigned int *vao, unsigned int *vbo) override;
        void renderScreenQuad_impl(unsigned int vao) override;
        //void createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color) override;
        //void createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color,
        //                               unsigned int *depth) override;
        //void destroyFramebuffer_impl(unsigned int *fbo, unsigned int *color, unsigned int *depth) override;
        void bindVertexArray_impl(unsigned int vao) override;
        void unbindVertexArray_impl() override;
        //void createTargets_impl(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples,
        //                           unsigned int *queryId0, unsigned int *queryId1) override;
        //void destroyTargets_impl(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1) override;
        //void resizeTargets_impl(CameraTargets *targets, Viewport viewport, bool *viewportChanged) override;
        void readColorAtPixel_impl(const unsigned int *fbo, int x, int y, Color32 *color) override;
        //void createTargets_impl(LightTargets *targets, ShadowMapResolution resolution) override;
        //void destroyTargets_impl(LightTargets *targets) override;
        //void resizeTargets_impl(LightTargets *targets, ShadowMapResolution resolution) override;
        
        //void createTexture2D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
        //                             int width, int height, const std::vector<unsigned char> &data,
        //                             TextureHandle*tex /*unsigned int* tex*/) override;
        //void destroyTexture2D_impl(TextureHandle*tex /*unsigned int* tex*/) override;
        //void updateTexture2D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
        //    TextureHandle* tex /*unsigned int tex*/) override;
        //void readPixelsTexture2D_impl(TextureFormat format, int width, int height, int numChannels,
        //                                 std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) override;
        //void writePixelsTexture2D_impl(TextureFormat format, int width, int height,
        //                                  const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) override;
        //void createTexture3D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
        //                             int width, int height, int depth, const std::vector<unsigned char> &data,
        //    TextureHandle* tex /*unsigned int* tex*/) override;
        //void destroyTexture3D_impl(TextureHandle* tex /*unsigned int* tex*/) override;
        //void updateTexture3D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
        //    TextureHandle* tex /*unsigned int tex*/) override;
        //void readPixelsTexture3D_impl(TextureFormat format, int width, int height, int depth, int numChannels,
        //                                 std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) override;
        //void writePixelsTexture3D_impl(TextureFormat format, int width, int height, int depth,
        //                                  const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) override;
        //void createCubemap_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
        //                           int width, const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/) override;
        //void destroyCubemap_impl(TextureHandle* tex /*unsigned int* tex*/) override;
        //void updateCubemap_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
        //    TextureHandle* tex /*unsigned int tex*/) override;
        //void readPixelsCubemap_impl(TextureFormat format, int width, int numChannels,
        //                               std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) override;
        //void writePixelsCubemap_impl(TextureFormat format, int width, const std::vector<unsigned char> &data,
        //    TextureHandle* tex /*unsigned int tex*/) override;
        
        
        //void createRenderTextureTargets_impl(RenderTextureTargets *targets, TextureFormat format,
        //                                        TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
        //                                        int height) override;
        //void destroyRenderTextureTargets_impl(RenderTextureTargets *targets) override;
        void createTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                        const std::vector<float> &texCoords, int vertexCount, unsigned int *vao,
                                        unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2) override;
        void destroyTerrainChunk_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1,
                                         unsigned int *vbo2) override;
        void updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                        unsigned int vbo0, unsigned int vbo1) override;
        void updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                        const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                                        unsigned int vbo2) override;
        void createMesh_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                const std::vector<float> &texCoords, unsigned int *vao, VertexBuffer*vbo0,
            VertexBuffer*vbo1, VertexBuffer*vbo2, VertexBuffer*model_vbo,
            VertexBuffer*color_vbo) override;
        void destroyMesh_impl(unsigned int *vao, VertexBuffer*vbo0, VertexBuffer*vbo1, VertexBuffer*vbo2,
            VertexBuffer*model_vbo, VertexBuffer*color_vbo) override;
        void updateInstanceBuffer_impl(unsigned int vbo, const glm::mat4 *models, size_t instanceCount) override;
        void updateInstanceColorBuffer_impl(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount) override;
        void createSprite_impl(unsigned int *vao) override;
        void destroySprite_impl(unsigned int *vao) override;
        void createFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                   unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1) override;
        void destroyFrustum_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1) override;
        void updateFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                                   unsigned int vbo0, unsigned int vbo1) override;
        void updateFrustum_impl(const std::vector<float> &vertices, unsigned int vbo0) override;
        void createGrid_impl(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0) override;
        void destroyGrid_impl(unsigned int *vao, unsigned int *vbo0) override;
        void createLine_impl(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                                unsigned int *vbo0, unsigned int *vbo1) override;
        void destroyLine_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1) override;

        void preprocess_impl(std::string &vert, std::string &frag, std::string &geom, int64_t variant) override;
        void compile_impl(const std::string &name, const std::string &vert, const std::string &frag,
                             const std::string &geom, unsigned int *program, ShaderStatus &status) override;
        int findUniformLocation_impl(const char *name, int program) override;
        int getUniformCount_impl(int program) override;
        int getAttributeCount_impl(int program) override;
        std::vector<ShaderUniform> getShaderUniforms_impl(int program) override;
        std::vector<ShaderAttribute> getShaderAttributes_impl(int program) override;
        void setUniformBlock_impl(const char *blockName, int bindingPoint, int program) override;
        /*void use_impl(int program) override;
        void unuse_impl() override;
        void destroy_impl(int program) override;
        void setBool_impl(int nameLocation, bool value) override;
        void setInt_impl(int nameLocation, int value) override;
        void setFloat_impl(int nameLocation, float value) override;
        void setColor_impl(int nameLocation, const Color &color) override;
        void setColor32_impl(int nameLocation, const Color32 &color) override;
        void setVec2_impl(int nameLocation, const glm::vec2 &vec) override;
        void setVec3_impl(int nameLocation, const glm::vec3 &vec) override;
        void setVec4_impl(int nameLocation, const glm::vec4 &vec) override;
        void setMat2_impl(int nameLocation, const glm::mat2 &mat) override;
        void setMat3_impl(int nameLocation, const glm::mat3 &mat) override;
        void setMat4_impl(int nameLocation, const glm::mat4 &mat) override;
        void setTexture2D_impl(int nameLocation, int texUnit, TextureHandle* tex) override;
        void setTexture2Ds_impl(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs) override;
        bool getBool_impl(int nameLocation, int program) override;
        int getInt_impl(int nameLocation, int program) override;
        float getFloat_impl(int nameLocation, int program) override;
        Color getColor_impl(int nameLocation, int program) override;
        Color32 getColor32_impl(int nameLocation, int program) override;
        glm::vec2 getVec2_impl(int nameLocation, int program) override;
        glm::vec3 getVec3_impl(int nameLocation, int program) override;
        glm::vec4 getVec4_impl(int nameLocation, int program) override;
        glm::mat2 getMat2_impl(int nameLocation, int program) override;
        glm::mat3 getMat3_impl(int nameLocation, int program) override;
        glm::mat4 getMat4_impl(int nameLocation, int program) override;*/
        void applyMaterial_impl(const std::vector<ShaderUniform> &uniforms, ShaderProgram* shaderProgram) override;
        void renderLines_impl(int start, int count, int vao) override;
        void renderLinesWithCurrentlyBoundVAO_impl(int start, int count) override;
        void renderWithCurrentlyBoundVAO_impl(int start, int count) override;
        void render_impl(int start, int count, int vao, bool wireframe = false) override;
        void render_impl(int start, int count, int vao, GraphicsQuery &query, bool wireframe = false) override;
        void renderInstanced_impl(int start, int count, int instanceCount, int vao, GraphicsQuery &query) override;
        void render_impl(const RenderObject &renderObject, GraphicsQuery &query) override;
        void renderInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query) override;
	};
}

#endif