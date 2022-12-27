#ifndef RENDERER_API_H__
#define RENDERER_API_H__

#include <string>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../components/Camera.h"
#include "../components/Light.h"
#include "../core/RenderTexture.h"
#include "../core/Shader.h"
#include "../core/Texture.h"

#include "GraphicsQuery.h"
#include "RenderObject.h"

#include "TextureHandle.h"
#include "VertexBuffer.h"
#include "UniformBuffer.h"

namespace PhysicsEngine
{
enum class Capability
{
    Depth_Testing,
    Blending,
    BackfaceCulling,
    LineSmoothing
};

enum class BlendingFactor
{
    ZERO,
    ONE,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA
};

class Renderer
{
private:
    static Renderer* sInstance;

public:
    static int INSTANCE_BATCH_SIZE;

    static void init();
    static Renderer *getRenderer();
    static void present();
    static void turnVsyncOn();
    static void turnVsyncOff();
    static void bindFramebuffer(Framebuffer* fbo);
    static void unbindFramebuffer();
    static void clearFrambufferColor(const Color &color);
    static void clearFrambufferColor(float r, float g, float b, float a);
    static void clearFramebufferDepth(float depth);
    static void setViewport(int x, int y, int width, int height);

    static void turnOn(Capability capability);
    static void turnOff(Capability capability);
    static void setBlending(BlendingFactor source, BlendingFactor dest);
    static void beginQuery(unsigned int queryId);
    static void endQuery(unsigned int queryId, unsigned long long *elapsedTime);
    //static void createGlobalCameraUniforms(CameraUniform &uniform);
    //static void createGlobalLightUniforms(LightUniform &uniform);
    //static void setGlobalCameraUniforms(const CameraUniform &uniform);
    //static void setGlobalLightUniforms(const LightUniform &uniform);
    static void createScreenQuad(unsigned int *vao, unsigned int *vbo);
    static void renderScreenQuad(unsigned int vao);
    //static void createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color);
    //static void createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color,
    //                               unsigned int *depth);
    //static void destroyFramebuffer(unsigned int *fbo, unsigned int *color, unsigned int *depth);
    static void bindVertexArray(unsigned int vao);
    static void unbindVertexArray();
    //static void createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples,
    //                           unsigned int *queryId0, unsigned int *queryId1);
    //static void destroyTargets(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1);
    //static void resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged);
    static void readColorAtPixel(const unsigned int *fbo, int x, int y, Color32 *color);
    //static void createTargets(LightTargets *targets, ShadowMapResolution resolution);
    //static void destroyTargets(LightTargets *targets);
    //static void resizeTargets(LightTargets *targets, ShadowMapResolution resolution);

  /*  static void createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
                                 int width, int height, const std::vector<unsigned char> &data, TextureHandle* tex);
    static void destroyTexture2D(TextureHandle*tex);
    static void updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
                                 TextureHandle*tex);
    static void readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                     std::vector<unsigned char> &data, TextureHandle* tex);
    static void writePixelsTexture2D(TextureFormat format, int width, int height,
                                      const std::vector<unsigned char> &data, TextureHandle* tex);
    static void createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode,
                                 int width, int height, int depth, const std::vector<unsigned char> &data,
                                 TextureHandle*tex);
    static void destroyTexture3D(TextureHandle* tex);
    static void updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
                                TextureHandle* tex);
    static void readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                     std::vector<unsigned char> &data, TextureHandle* tex);
    static void writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                      const std::vector<unsigned char> &data, TextureHandle* tex);
    static void createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               const std::vector<unsigned char> &data, TextureHandle* tex);
    static void destroyCubemap(TextureHandle* tex);
    static void updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
        TextureHandle* tex);
    static void readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
        TextureHandle* tex);
    static void writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data,
        TextureHandle* tex);*/
    //static void createRenderTextureTargets(RenderTextureTargets *targets, TextureFormat format,
    //                                        TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
    //                                        int height);
    //static void destroyRenderTextureTargets(RenderTextureTargets *targets);
    static void createTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                    const std::vector<float> &texCoords, int vertexCount, unsigned int *vao,
                                    unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2);
    static void destroyTerrainChunk(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2);
    static void updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                    unsigned int vbo0, unsigned int vbo1);
    static void updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                    const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                                    unsigned int vbo2);
    /*static void createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                            const std::vector<float> &texCoords, unsigned int *vao, VertexBuffer *vbo0,
        VertexBuffer*vbo1, VertexBuffer*vbo2, VertexBuffer*model_vbo,
        VertexBuffer*color_vbo);
    static void destroyMesh(unsigned int *vao, VertexBuffer*vbo0, VertexBuffer*vbo1, VertexBuffer*vbo2,
        VertexBuffer*model_vbo, VertexBuffer*color_vbo);*/
    static void updateInstanceBuffer(unsigned int vbo, const glm::mat4 *models, size_t instanceCount);
    static void updateInstanceColorBuffer(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount);
    static void createSprite(unsigned int *vao);
    static void destroySprite(unsigned int *vao);
    static void createFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int *vao,
                               unsigned int *vbo0, unsigned int *vbo1);
    static void destroyFrustum(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1);
    static void updateFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int vbo0,
                               unsigned int vbo1);
    static void updateFrustum(const std::vector<float> &vertices, unsigned int vbo0);
    static void createGrid(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0);
    static void destroyGrid(unsigned int *vao, unsigned int *vbo0);
    static void createLine(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                            unsigned int *vbo0, unsigned int *vbo1);
    static void destroyLine(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1);
    /*static void preprocess(std::string &vert, std::string &frag, std::string &geom, int64_t variant);
    static void compile(const std::string &name, const std::string &vert, const std::string &frag,
                         const std::string &geom, unsigned int *program, ShaderStatus &status);
    static int findUniformLocation(const char *name, int program);
    static int getUniformCount(int program);
    static int getAttributeCount(int program);
    static std::vector<ShaderUniform> getShaderUniforms(int program);
    static std::vector<ShaderAttribute> getShaderAttributes(int program);
    static void setUniformBlock(const char *blockName, int bindingPoint, int program);*/
    /*static void use(int program);
    static void unuse();
    static void destroy(int program);
    static void setBool(int nameLocation, bool value);
    static void setInt(int nameLocation, int value);
    static void setFloat(int nameLocation, float value);
    static void setColor(int nameLocation, const Color &color);
    static void setColor32(int nameLocation, const Color32 &color);
    static void setVec2(int nameLocation, const glm::vec2 &vec);
    static void setVec3(int nameLocation, const glm::vec3 &vec);
    static void setVec4(int nameLocation, const glm::vec4 &vec);
    static void setMat2(int nameLocation, const glm::mat2 &mat);
    static void setMat3(int nameLocation, const glm::mat3 &mat);
    static void setMat4(int nameLocation, const glm::mat4 &mat);
    static void setTexture2D(int nameLocation, int texUnit, TextureHandle* tex);
    static void setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs);
    static bool getBool(int nameLocation, int program);
    static int getInt(int nameLocation, int program);
    static float getFloat(int nameLocation, int program);
    static Color getColor(int nameLocation, int program);
    static Color32 getColor32(int nameLocation, int program);
    static glm::vec2 getVec2(int nameLocation, int program);
    static glm::vec3 getVec3(int nameLocation, int program);
    static glm::vec4 getVec4(int nameLocation, int program);
    static glm::mat2 getMat2(int nameLocation, int program);
    static glm::mat3 getMat3(int nameLocation, int program);
    static glm::mat4 getMat4(int nameLocation, int program);*/
    //static void applyMaterial(const std::vector<ShaderUniform> &uniforms, ShaderProgram* shaderProgram);
    static void renderLines(int start, int count, int vao);
    static void renderLinesWithCurrentlyBoundVAO(int start, int count);
    static void renderWithCurrentlyBoundVAO(int start, int count);
    static void render(int start, int count, int vao, bool wireframe = false);
    static void render(int start, int count, int vao, GraphicsQuery &query, bool wireframe = false);
    static void renderInstanced(int start, int count, int instanceCount, int vao, GraphicsQuery &query);
    static void render(const RenderObject &renderObject, GraphicsQuery &query);
    static void renderInstanced(const RenderObject &renderObject, GraphicsQuery &query);
    
protected:
    virtual void init_impl() = 0;
    virtual void present_impl() = 0;
    virtual void turnVsyncOn_impl() = 0;
    virtual void turnVsyncOff_impl() = 0;
    virtual void bindFramebuffer_impl(Framebuffer* fbo) = 0;
    virtual void unbindFramebuffer_impl() = 0;
    virtual void clearFrambufferColor_impl(const Color &color) = 0;
    virtual void clearFrambufferColor_impl(float r, float g, float b, float a) = 0;
    virtual void clearFramebufferDepth_impl(float depth) = 0;
    virtual void setViewport_impl(int x, int y, int width, int height) = 0;

    virtual void turnOn_impl(Capability capability) = 0;
    virtual void turnOff_impl(Capability capability) = 0;
    virtual void setBlending_impl(BlendingFactor source, BlendingFactor dest) = 0;
    virtual void beginQuery_impl(unsigned int queryId) = 0;
    virtual void endQuery_impl(unsigned int queryId, unsigned long long* elapsedTime) = 0;
    //virtual void createGlobalCameraUniforms_impl(CameraUniform& uniform) = 0;
    //virtual void createGlobalLightUniforms_impl(LightUniform& uniform) = 0;
    //virtual void setGlobalCameraUniforms_impl(const CameraUniform& uniform) = 0;
    //virtual void setGlobalLightUniforms_impl(const LightUniform& uniform) = 0;
    virtual void createScreenQuad_impl(unsigned int* vao, unsigned int* vbo) = 0;
    virtual void renderScreenQuad_impl(unsigned int vao) = 0;
    //virtual void createFramebuffer_impl(int width, int height, unsigned int* fbo, unsigned int* color) = 0;
    //virtual void createFramebuffer_impl(int width, int height, unsigned int* fbo, unsigned int* color, unsigned int* depth) = 0;
    //virtual void destroyFramebuffer_impl(unsigned int* fbo, unsigned int* color, unsigned int* depth) = 0;
    virtual void bindVertexArray_impl(unsigned int vao) = 0;
    virtual void unbindVertexArray_impl() = 0;
    //virtual void createTargets_impl(CameraTargets* targets, Viewport viewport, glm::vec3* ssaoSamples, unsigned int* queryId0, unsigned int* queryId1) = 0;
    //virtual void destroyTargets_impl(CameraTargets* targets, unsigned int* queryId0, unsigned int* queryId1) = 0;
    //virtual void resizeTargets_impl(CameraTargets* targets, Viewport viewport, bool* viewportChanged) = 0;
    virtual void readColorAtPixel_impl(const unsigned int* fbo, int x, int y, Color32* color) = 0;
    //virtual void createTargets_impl(LightTargets* targets, ShadowMapResolution resolution) = 0;
    //virtual void destroyTargets_impl(LightTargets* targets) = 0;
    //virtual void resizeTargets_impl(LightTargets* targets, ShadowMapResolution resolution) = 0;
    //virtual void createTexture2D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
    //    int height, const std::vector<unsigned char>& data, TextureHandle*tex /*unsigned int* tex*/) = 0;
    //virtual void destroyTexture2D_impl(TextureHandle*tex /*unsigned int* tex*/) = 0;
    //virtual void updateTexture2D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void readPixelsTexture2D_impl(TextureFormat format, int width, int height, int numChannels,
    //    std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void writePixelsTexture2D_impl(TextureFormat format, int width, int height,
    //    const std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void createTexture3D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
    //    int height, int depth, const std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int* tex*/) = 0;
    //virtual void destroyTexture3D_impl(TextureHandle* tex /*unsigned int* tex*/) = 0;
    //virtual void updateTexture3D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void readPixelsTexture3D_impl(TextureFormat format, int width, int height, int depth, int numChannels,
    //    std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void writePixelsTexture3D_impl(TextureFormat format, int width, int height, int depth,
    //    const std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void createCubemap_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
    //    const std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int* tex*/) = 0;
    //virtual void destroyCubemap_impl(TextureHandle* tex /*unsigned int* tex*/) = 0;
    //virtual void updateCubemap_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void readPixelsCubemap_impl(TextureFormat format, int width, int numChannels, std::vector<unsigned char>& data,
    //    TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void writePixelsCubemap_impl(TextureFormat format, int width, const std::vector<unsigned char>& data, TextureHandle* tex /*unsigned int tex*/) = 0;
    //virtual void createRenderTextureTargets_impl(RenderTextureTargets* targets, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, int height) = 0;
    //virtual void destroyRenderTextureTargets_impl(RenderTextureTargets* targets) = 0;
    virtual void createTerrainChunk_impl(const std::vector<float>& vertices, const std::vector<float>& normals,
        const std::vector<float>& texCoords, int vertexCount,
        unsigned int* vao, unsigned int* vbo0, unsigned int* vbo1, unsigned int* vbo2) = 0;
    virtual void destroyTerrainChunk_impl(unsigned int* vao, unsigned int* vbo0, unsigned int* vbo1, unsigned int* vbo2) = 0;
    virtual void updateTerrainChunk_impl(const std::vector<float>& vertices, const std::vector<float>& normals,
        unsigned int vbo0, unsigned int vbo1) = 0;
    virtual void updateTerrainChunk_impl(const std::vector<float>& vertices, const std::vector<float>& normals,
        const std::vector<float>& texCoords, unsigned int vbo0, unsigned int vbo1,
        unsigned int vbo2) = 0;
    /*virtual void createMesh_impl(const std::vector<float>& vertices, const std::vector<float>& normals,
        const std::vector<float>& texCoords, unsigned int* vao, VertexBuffer* vbo0, VertexBuffer* vbo1, VertexBuffer* vbo2, VertexBuffer* model_vbo, VertexBuffer* color_vbo) = 0;*/
    //virtual void destroyMesh_impl(unsigned int* vao, VertexBuffer* vbo0, VertexBuffer* vbo1, VertexBuffer* vbo2, VertexBuffer* model_vbo, VertexBuffer* color_vbo) = 0;
    virtual void updateInstanceBuffer_impl(unsigned int vbo, const glm::mat4* models, size_t instanceCount) = 0;
    virtual void updateInstanceColorBuffer_impl(unsigned int vbo, const glm::vec4* colors, size_t instanceCount) = 0;
    virtual void createSprite_impl(unsigned int* vao) = 0;
    virtual void destroySprite_impl(unsigned int* vao) = 0;
    virtual void createFrustum_impl(const std::vector<float>& vertices, const std::vector<float>& normals, unsigned int* vao,
        unsigned int* vbo0, unsigned int* vbo1) = 0;
    virtual void destroyFrustum_impl(unsigned int* vao, unsigned int* vbo0, unsigned int* vbo1) = 0;
    virtual void updateFrustum_impl(const std::vector<float>& vertices, const std::vector<float>& normals, unsigned int vbo0,
        unsigned int vbo1) = 0;
    virtual void updateFrustum_impl(const std::vector<float>& vertices, unsigned int vbo0) = 0;
    virtual void createGrid_impl(const std::vector<glm::vec3>& vertices, unsigned int* vao, unsigned int* vbo0) = 0;
    virtual void destroyGrid_impl(unsigned int* vao, unsigned int* vbo0) = 0;
    virtual void createLine_impl(const std::vector<float>& vertices, const std::vector<float>& colors, unsigned int* vao,
        unsigned int* vbo0, unsigned int* vbo1) = 0;
    virtual void destroyLine_impl(unsigned int* vao, unsigned int* vbo0, unsigned int* vbo1) = 0;
    /*virtual void preprocess_impl(std::string& vert, std::string& frag, std::string& geom, int64_t variant) = 0;
    virtual void compile_impl(const std::string& name, const std::string& vert, const std::string& frag,
        const std::string& geom, unsigned int* program, ShaderStatus& status) = 0;
    virtual int findUniformLocation_impl(const char* name, int program) = 0;
    virtual int getUniformCount_impl(int program) = 0;
    virtual int getAttributeCount_impl(int program) = 0;
    virtual std::vector<ShaderUniform> getShaderUniforms_impl(int program) = 0;
    virtual std::vector<ShaderAttribute> getShaderAttributes_impl(int program) = 0;
    virtual void setUniformBlock_impl(const char* blockName, int bindingPoint, int program) = 0;*/
    /*virtual void use_impl(int program) = 0;
    virtual void unuse_impl() = 0;
    virtual void destroy_impl(int program) = 0;
    virtual void setBool_impl(int nameLocation, bool value) = 0;
    virtual void setInt_impl(int nameLocation, int value) = 0;
    virtual void setFloat_impl(int nameLocation, float value) = 0;
    virtual void setColor_impl(int nameLocation, const Color& color) = 0;
    virtual void setColor32_impl(int nameLocation, const Color32& color) = 0;
    virtual void setVec2_impl(int nameLocation, const glm::vec2& vec) = 0;
    virtual void setVec3_impl(int nameLocation, const glm::vec3& vec) = 0;
    virtual void setVec4_impl(int nameLocation, const glm::vec4& vec) = 0;
    virtual void setMat2_impl(int nameLocation, const glm::mat2& mat) = 0;
    virtual void setMat3_impl(int nameLocation, const glm::mat3& mat) = 0;
    virtual void setMat4_impl(int nameLocation, const glm::mat4& mat) = 0;
    virtual void setTexture2D_impl(int nameLocation, int texUnit, TextureHandle* tex) = 0;
    virtual void setTexture2Ds_impl(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs) = 0;
    virtual bool getBool_impl(int nameLocation, int program) = 0;
    virtual int getInt_impl(int nameLocation, int program) = 0;
    virtual float getFloat_impl(int nameLocation, int program) = 0;
    virtual Color getColor_impl(int nameLocation, int program) = 0;
    virtual Color32 getColor32_impl(int nameLocation, int program) = 0;
    virtual glm::vec2 getVec2_impl(int nameLocation, int program) = 0;
    virtual glm::vec3 getVec3_impl(int nameLocation, int program) = 0;
    virtual glm::vec4 getVec4_impl(int nameLocation, int program) = 0;
    virtual glm::mat2 getMat2_impl(int nameLocation, int program) = 0;
    virtual glm::mat3 getMat3_impl(int nameLocation, int program) = 0;
    virtual glm::mat4 getMat4_impl(int nameLocation, int program) = 0;*/
    //virtual void applyMaterial_impl(const std::vector<ShaderUniform>& uniforms, ShaderProgram* shaderProgram) = 0;
    virtual void renderLines_impl(int start, int count, int vao) = 0;
    virtual void renderLinesWithCurrentlyBoundVAO_impl(int start, int count) = 0;
    virtual void renderWithCurrentlyBoundVAO_impl(int start, int count) = 0;
    virtual void render_impl(int start, int count, int vao, bool wireframe = false) = 0;
    virtual void render_impl(int start, int count, int vao, GraphicsQuery& query, bool wireframe = false) = 0;
    virtual void renderInstanced_impl(int start, int count, int instanceCount, int vao, GraphicsQuery& query) = 0;
    virtual void render_impl(const RenderObject& renderObject, GraphicsQuery& query) = 0;
    virtual void renderInstanced_impl(const RenderObject& renderObject, GraphicsQuery& query) = 0;
};
}

#endif // RENDERER_API_H__