#include <algorithm>
#include <assert.h>
#include <iostream>
#include <random>

#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXRenderer.h"

using namespace PhysicsEngine;

#define CHECK_ERROR_IMPL(ROUTINE, LINE, FILE)           \
    do{                                                 \
        HRESULT hr = ROUTINE;                           \
        LPTSTR lpBuf = NULL;                            \
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |  \
                      FORMAT_MESSAGE_FROM_SYSTEM |      \
                      FORMAT_MESSAGE_IGNORE_INSERTS,    \
                      NULL,                             \
                      hr,                               \
                      0,                                \
                      (LPTSTR)&lpBuf,                   \
                      0,                                \
                      NULL);                            \
    }while(0)

#define CHECK_ERROR(ROUTINE) CHECK_ERROR_IMPL(ROUTINE, std::to_string(__LINE__), std::string(__FILE__))

void DirectXRenderer::init_impl()
{
    mContext = DirectXRenderContext::get();

    DXGI_ADAPTER_DESC descr;
    mContext->getAdapter()->GetDesc(&descr);

    //Log::warn(("Vender: " + vender + "\n").c_str());
    Log::warn(("Dedicated video memory: " + std::to_string(descr.DedicatedVideoMemory) + "\n").c_str());
}
void DirectXRenderer::present_impl()
{
    mContext->present();
}

void DirectXRenderer::turnVsyncOn_impl()
{
    mContext->turnVsyncOn();
}

void DirectXRenderer::turnVsyncOff_impl()
{
    mContext->turnVsyncOff();
}

void DirectXRenderer::turnOn_impl(Capability capability){}
void DirectXRenderer::turnOff_impl(Capability capability){}
void DirectXRenderer::setBlending_impl(BlendingFactor source, BlendingFactor dest){}
void DirectXRenderer::beginQuery_impl(unsigned int queryId){}
void DirectXRenderer::endQuery_impl(unsigned int queryId, unsigned long long *elapsedTime){}
void DirectXRenderer::createGlobalCameraUniforms_impl(CameraUniform &uniform){}
void DirectXRenderer::createGlobalLightUniforms_impl(LightUniform &uniform){}
void DirectXRenderer::setGlobalCameraUniforms_impl(const CameraUniform &uniform){}
void DirectXRenderer::setGlobalLightUniforms_impl(const LightUniform &uniform){}
void DirectXRenderer::createScreenQuad_impl(unsigned int *vao, unsigned int *vbo){}
void DirectXRenderer::renderScreenQuad_impl(unsigned int vao){}
//void DirectXRenderer::createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color){}
//void DirectXRenderer::createFramebuffer_impl(int width, int height, unsigned int *fbo, unsigned int *color,
//                            unsigned int *depth){}
//void DirectXRenderer::destroyFramebuffer_impl(unsigned int *fbo, unsigned int *color, unsigned int *depth){}
void DirectXRenderer::bindFramebuffer_impl(unsigned int fbo){}
void DirectXRenderer::unbindFramebuffer_impl(){}
void DirectXRenderer::clearFrambufferColor_impl(const Color &color){}
void DirectXRenderer::clearFrambufferColor_impl(float r, float g, float b, float a){}
void DirectXRenderer::clearFramebufferDepth_impl(float depth){}
void DirectXRenderer::bindVertexArray_impl(unsigned int vao){}
void DirectXRenderer::unbindVertexArray_impl(){}
void DirectXRenderer::setViewport_impl(int x, int y, int width, int height){}
//void DirectXRenderer::createTargets_impl(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, unsigned int *queryId0,
//                        unsigned int *queryId1){}
//void DirectXRenderer::destroyTargets_impl(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1){}
//void DirectXRenderer::resizeTargets_impl(CameraTargets *targets, Viewport viewport, bool *viewportChanged){}
void DirectXRenderer::readColorAtPixel_impl(const unsigned int *fbo, int x, int y, Color32 *color){}
//void DirectXRenderer::createTargets_impl(LightTargets *targets, ShadowMapResolution resolution){}
//void DirectXRenderer::destroyTargets_impl(LightTargets *targets){}
//void DirectXRenderer::resizeTargets_impl(LightTargets *targets, ShadowMapResolution resolution){}
//void DirectXRenderer::createTexture2D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
//                          int height, const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::destroyTexture2D_impl(TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::updateTexture2D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::readPixelsTexture2D_impl(TextureFormat format, int width, int height, int numChannels,
//                              std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::writePixelsTexture2D_impl(TextureFormat format, int width, int height, const std::vector<unsigned char> &data,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::createTexture3D_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
//                          int height, int depth, const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::destroyTexture3D_impl(TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::updateTexture3D_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::readPixelsTexture3D_impl(TextureFormat format, int width, int height, int depth, int numChannels,
//                              std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::writePixelsTexture3D_impl(TextureFormat format, int width, int height, int depth,
//                               const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::createCubemap_impl(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
//                        const std::vector<unsigned char> &data, TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::destroyCubemap_impl(TextureHandle* tex /*unsigned int* tex*/) {}
//void DirectXRenderer::updateCubemap_impl(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::readPixelsCubemap_impl(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::writePixelsCubemap_impl(TextureFormat format, int width, const std::vector<unsigned char> &data,
//    TextureHandle* tex /*unsigned int tex*/) {}
//void DirectXRenderer::createRenderTextureTargets_impl(RenderTextureTargets *targets, TextureFormat format, TextureWrapMode wrapMode,
//                                     TextureFilterMode filterMode, int width, int height){}
//void DirectXRenderer::destroyRenderTextureTargets_impl(RenderTextureTargets *targets){}
void DirectXRenderer::createTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                             const std::vector<float> &texCoords, int vertexCount, unsigned int *vao,
                             unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2){}
void DirectXRenderer::destroyTerrainChunk_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2){}
void DirectXRenderer::updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int vbo0,
                             unsigned int vbo1){}
void DirectXRenderer::updateTerrainChunk_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                             const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                             unsigned int vbo2){}
void DirectXRenderer::createMesh_impl(const std::vector<float> &vertices, const std::vector<float> &normals,
                     const std::vector<float> &texCoords, unsigned int *vao, VertexBuffer*vbo0, VertexBuffer*vbo1,
    VertexBuffer*vbo2, VertexBuffer*model_vbo, VertexBuffer*color_vbo){}
void DirectXRenderer::destroyMesh_impl(unsigned int *vao, VertexBuffer*vbo0, VertexBuffer*vbo1, VertexBuffer*vbo2,
    VertexBuffer*model_vbo, VertexBuffer*color_vbo){}
void DirectXRenderer::updateInstanceBuffer_impl(unsigned int vbo, const glm::mat4 *models, size_t instanceCount){}
void DirectXRenderer::updateInstanceColorBuffer_impl(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount){}
void DirectXRenderer::createSprite_impl(unsigned int *vao){}
void DirectXRenderer::destroySprite_impl(unsigned int *vao){}
void DirectXRenderer::createFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int *vao,
                        unsigned int *vbo0, unsigned int *vbo1){}
void DirectXRenderer::destroyFrustum_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1){}
void DirectXRenderer::updateFrustum_impl(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int vbo0,
                        unsigned int vbo1){}
void DirectXRenderer::updateFrustum_impl(const std::vector<float> &vertices, unsigned int vbo0){}
void DirectXRenderer::createGrid_impl(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0){}
void DirectXRenderer::destroyGrid_impl(unsigned int *vao, unsigned int *vbo0){}
void DirectXRenderer::createLine_impl(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                     unsigned int *vbo0, unsigned int *vbo1){}
void DirectXRenderer::destroyLine_impl(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1){}

void DirectXRenderer::preprocess_impl(std::string &vert, std::string &frag, std::string &geom, int64_t variant){}
void DirectXRenderer::compile_impl(const std::string &name, const std::string &vert, const std::string &frag, const std::string &geom,
                  unsigned int *program, ShaderStatus &status){}
int DirectXRenderer::findUniformLocation_impl(const char *name, int program)
{
    return 0;
}
int DirectXRenderer::getUniformCount_impl(int program)
{
    return 0;
}
int DirectXRenderer::getAttributeCount_impl(int program)
{
    return 0;
}
std::vector<ShaderUniform> DirectXRenderer::getShaderUniforms_impl(int program)
{
    return std::vector<ShaderUniform>();
}
std::vector<ShaderAttribute> DirectXRenderer::getShaderAttributes_impl(int program)
{
    return std::vector<ShaderAttribute>();
}
void DirectXRenderer::setUniformBlock_impl(const char *blockName, int bindingPoint, int program){}
void DirectXRenderer::use_impl(int program){}
void DirectXRenderer::unuse_impl(){}
void DirectXRenderer::destroy_impl(int program){}
void DirectXRenderer::setBool_impl(int nameLocation, bool value){}
void DirectXRenderer::setInt_impl(int nameLocation, int value){}
void DirectXRenderer::setFloat_impl(int nameLocation, float value){}
void DirectXRenderer::setColor_impl(int nameLocation, const Color &color){}
void DirectXRenderer::setColor32_impl(int nameLocation, const Color32 &color){}
void DirectXRenderer::setVec2_impl(int nameLocation, const glm::vec2 &vec){}
void DirectXRenderer::setVec3_impl(int nameLocation, const glm::vec3 &vec){}
void DirectXRenderer::setVec4_impl(int nameLocation, const glm::vec4 &vec){}
void DirectXRenderer::setMat2_impl(int nameLocation, const glm::mat2 &mat){}
void DirectXRenderer::setMat3_impl(int nameLocation, const glm::mat3 &mat){}
void DirectXRenderer::setMat4_impl(int nameLocation, const glm::mat4 &mat){}
void DirectXRenderer::setTexture2D_impl(int nameLocation, int texUnit, TextureHandle* tex){}
void DirectXRenderer::setTexture2Ds_impl(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs){}
bool DirectXRenderer::getBool_impl(int nameLocation, int program)
{
    return false;
}

int DirectXRenderer::getInt_impl(int nameLocation, int program)
{
    return 0;
}

float DirectXRenderer::getFloat_impl(int nameLocation, int program)
{
    return 0.0f;
}

Color DirectXRenderer::getColor_impl(int nameLocation, int program)
{
    return Color(0.0f, 0.0f, 0.0f, 1.0f);
}

Color32 DirectXRenderer::getColor32_impl(int nameLocation, int program)
{
    return Color32(0, 0, 0, 255);
}

glm::vec2 DirectXRenderer::getVec2_impl(int nameLocation, int program)
{
    return glm::vec2();
}
glm::vec3 DirectXRenderer::getVec3_impl(int nameLocation, int program)
{
    return glm::vec3();
}
glm::vec4 DirectXRenderer::getVec4_impl(int nameLocation, int program)
{
    return glm::vec4();
}
glm::mat2 DirectXRenderer::getMat2_impl(int nameLocation, int program)
{
    return glm::mat2();
}
glm::mat3 DirectXRenderer::getMat3_impl(int nameLocation, int program)
{
    return glm::mat3();
}
glm::mat4 DirectXRenderer::getMat4_impl(int nameLocation, int program)
{
    return glm::mat4();
}
int DirectXRenderer::getTexture2D_impl(int nameLocation, int texUnit, int program)
{
    return 0;
}
void DirectXRenderer::applyMaterial_impl(const std::vector<ShaderUniform> &uniforms, int shaderProgram){}
void DirectXRenderer::renderLines_impl(int start, int count, int vao){}
void DirectXRenderer::renderLinesWithCurrentlyBoundVAO_impl(int start, int count){}
void DirectXRenderer::renderWithCurrentlyBoundVAO_impl(int start, int count){}
void DirectXRenderer::render_impl(int start, int count, int vao, bool wireframe){}
void DirectXRenderer::render_impl(int start, int count, int vao, GraphicsQuery &query, bool wireframe){}
void DirectXRenderer::renderInstanced_impl(int start, int count, int instanceCount, int vao, GraphicsQuery &query){}
void DirectXRenderer::render_impl(const RenderObject &renderObject, GraphicsQuery &query){}
void DirectXRenderer::renderInstanced_impl(const RenderObject &renderObject, GraphicsQuery &query){}