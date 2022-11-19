#include "../../include/graphics/Renderer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../include/graphics/platform/directx/DirectXRenderer.h"

using namespace PhysicsEngine;

int Renderer::INSTANCE_BATCH_SIZE = 1000;
Renderer *Renderer::sInstance = nullptr;

void Renderer::init()
{
    switch (RenderContext::getRenderAPI())
    {
        case RenderAPI::OpenGL:
            sInstance = new OpenGLRenderer();
            break;
        case RenderAPI::DirectX:
            sInstance = new DirectXRenderer();
            break;
    }

    sInstance->init_impl();
}

Renderer *Renderer::getRenderer()
{
    return sInstance;
}

void Renderer::present()
{
    sInstance->present_impl();
}

void Renderer::turnVsyncOn()
{
    sInstance->turnVsyncOn_impl();
}

void Renderer::turnVsyncOff()
{
    sInstance->turnVsyncOff_impl();
}
    
void Renderer::turnOn(Capability capability)
{
    return sInstance->turnOn_impl(capability);
}

void Renderer::turnOff(Capability capability)
{
    return sInstance->turnOff_impl(capability);
}

void Renderer::setBlending(BlendingFactor source, BlendingFactor dest)
{
    return sInstance->setBlending_impl(source, dest);
}

void Renderer::beginQuery(unsigned int queryId)
{
    return sInstance->beginQuery_impl(queryId);
}

void Renderer::endQuery(unsigned int queryId, unsigned long long *elapsedTime)
{
    return sInstance->endQuery_impl(queryId, elapsedTime);
}

void Renderer::createGlobalCameraUniforms(CameraUniform &uniform)
{
    return sInstance->createGlobalCameraUniforms_impl(uniform);
}

void Renderer::createGlobalLightUniforms(LightUniform &uniform)
{
    return sInstance->createGlobalLightUniforms_impl(uniform);
}

void Renderer::setGlobalCameraUniforms(const CameraUniform &uniform)
{
    return sInstance->setGlobalCameraUniforms_impl(uniform);
}

void Renderer::setGlobalLightUniforms(const LightUniform &uniform)
{
    return sInstance->setGlobalLightUniforms_impl(uniform);
}

void Renderer::createScreenQuad(unsigned int *vao, unsigned int *vbo)
{
    return sInstance->createScreenQuad_impl(vao, vbo);
}

void Renderer::renderScreenQuad(unsigned int vao)
{
    return sInstance->renderScreenQuad_impl(vao);
}

void Renderer::createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color)
{
    return sInstance->createFramebuffer_impl(width, height, fbo, color);
}

void Renderer::createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color, unsigned int *depth)
{
    return sInstance->createFramebuffer_impl(width, height, fbo, color, depth);
}

void Renderer::destroyFramebuffer(unsigned int *fbo, unsigned int *color, unsigned int *depth)
{
    return sInstance->destroyFramebuffer_impl(fbo, color, depth);
}

void Renderer::bindFramebuffer(unsigned int fbo)
{
    return sInstance->bindFramebuffer_impl(fbo);
}

void Renderer::unbindFramebuffer()
{
    return sInstance->unbindFramebuffer_impl();
}

void Renderer::clearFrambufferColor(const Color &color)
{
    return sInstance->clearFrambufferColor_impl(color);
}

void Renderer::clearFrambufferColor(float r, float g, float b, float a)
{
    return sInstance->clearFrambufferColor_impl(r, g, b, a);
}

void Renderer::clearFramebufferDepth(float depth)
{
    return sInstance->clearFramebufferDepth_impl(depth);
}

void Renderer::bindVertexArray(unsigned int vao)
{
    return sInstance->bindVertexArray_impl(vao);
}

void Renderer::unbindVertexArray()
{
    return sInstance->unbindVertexArray_impl();
}

void Renderer::setViewport(int x, int y, int width, int height)
{
    return sInstance->setViewport_impl(x, y, width, height);
}

void Renderer::createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, unsigned int *queryId0,
                            unsigned int *queryId1)
{
    return sInstance->createTargets_impl(targets, viewport, ssaoSamples, queryId0, queryId1);
}

void Renderer::destroyTargets(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1)
{
    return sInstance->destroyTargets_impl(targets, queryId0, queryId1);
}

void Renderer::resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged)
{
    return sInstance->resizeTargets_impl(targets, viewport, viewportChanged);
}

void Renderer::readColorAtPixel(const unsigned int *fbo, int x, int y, Color32 *color)
{
    return sInstance->readColorAtPixel_impl(fbo, x, y, color);
}

void Renderer::createTargets(LightTargets *targets, ShadowMapResolution resolution)
{
    return sInstance->createTargets_impl(targets, resolution);
}

void Renderer::destroyTargets(LightTargets *targets)
{
    return sInstance->destroyTargets_impl(targets);
}

void Renderer::resizeTargets(LightTargets *targets, ShadowMapResolution resolution)
{
    return sInstance->resizeTargets_impl(targets, resolution);
}

void Renderer::createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                            int height, const std::vector<unsigned char> &data, unsigned int *tex)
{
    return sInstance->createTexture2D_impl(format, wrapMode, filterMode, width, height, data, tex);
}

void Renderer::destroyTexture2D(unsigned int *tex)
{
    return sInstance->destroyTexture2D_impl(tex);
}

void Renderer::updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
                            unsigned int tex)
{
    return sInstance->updateTexture2D_impl(wrapMode, filterMode, anisoLevel, tex);
}

void Renderer::readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                std::vector<unsigned char> &data, unsigned int tex)
{
    return sInstance->readPixelsTexture2D_impl(format, width, height, numChannels, data, tex);
}

void Renderer::writePixelsTexture2D(TextureFormat format, int width, int height,
                                    const std::vector<unsigned char> &data, unsigned int tex)
{
    return sInstance->writePixelsTexture2D_impl(format, width, height, data, tex);
}

void Renderer::createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                            int height, int depth, const std::vector<unsigned char> &data, unsigned int *tex)
{
    return sInstance->createTexture3D_impl(format, wrapMode, filterMode, width, height, depth, data, tex);
}

void Renderer::destroyTexture3D(unsigned int *tex)
{
    return sInstance->destroyTexture3D_impl(tex);
}

void Renderer::updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel,
                            unsigned int tex)
{
    return sInstance->updateTexture3D_impl(wrapMode, filterMode, anisoLevel, tex);
}

void Renderer::readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                std::vector<unsigned char> &data, unsigned int tex)
{
    return sInstance->readPixelsTexture3D_impl(format, width, height, depth, numChannels, data, tex);
}

void Renderer::writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                    const std::vector<unsigned char> &data, unsigned int tex)
{
    return sInstance->writePixelsTexture3D_impl(format, width, height, depth, data, tex);
}

void Renderer::createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                            const std::vector<unsigned char> &data, unsigned int *tex)
{
    return sInstance->createCubemap_impl(format, wrapMode, filterMode, width, data, tex);
}

void Renderer::destroyCubemap(unsigned int *tex)
{
    return sInstance->destroyCubemap_impl(tex);
}

void Renderer::updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex)
{
    return sInstance->updateCubemap_impl(wrapMode, filterMode, anisoLevel, tex);
}

void Renderer::readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
                                unsigned int tex)
{
    return sInstance->readPixelsCubemap_impl(format, width, numChannels, data, tex);
}

void Renderer::writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data,
                                unsigned int tex)
{
    return sInstance->writePixelsCubemap_impl(format, width, data, tex);
}

void Renderer::createRenderTextureTargets(RenderTextureTargets *targets, TextureFormat format,
                                        TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                                        int height)
{
    return sInstance->createRenderTextureTargets_impl(targets, format, wrapMode, filterMode, width, height);
}

void Renderer::destroyRenderTextureTargets(RenderTextureTargets *targets)
{
    return sInstance->destroyRenderTextureTargets_impl(targets);
}

void Renderer::createTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                const std::vector<float> &texCoords, int vertexCount, unsigned int *vao,
                                unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2)
{
    return sInstance->createTerrainChunk_impl(vertices, normals, texCoords, vertexCount, vao,
                                            vbo0, vbo1, vbo2);
}

void Renderer::destroyTerrainChunk(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2)
{
    return sInstance->destroyTerrainChunk_impl(vao, vbo0, vbo1, vbo2);
}

void Renderer::updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                unsigned int vbo0, unsigned int vbo1)
{
    return sInstance->updateTerrainChunk_impl(vertices, normals, vbo0, vbo1);
}

void Renderer::updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1,
                                unsigned int vbo2)
{
    return sInstance->updateTerrainChunk_impl(vertices, normals, texCoords, vbo0, vbo1, vbo2);
}

void Renderer::createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                        const std::vector<float> &texCoords, unsigned int *vao, unsigned int *vbo0,
                        unsigned int *vbo1, unsigned int *vbo2, unsigned int *model_vbo, unsigned int *color_vbo)
{
    return sInstance->createMesh_impl(vertices, normals, texCoords, vao, vbo0, vbo1, vbo2, model_vbo, color_vbo);
}

void Renderer::destroyMesh(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2,
                        unsigned int *model_vbo, unsigned int *color_vbo)
{
    return sInstance->destroyMesh_impl(vao, vbo0, vbo1, vbo2, model_vbo, color_vbo);
}

void Renderer::updateInstanceBuffer(unsigned int vbo, const glm::mat4 *models, size_t instanceCount)
{
    return sInstance->updateInstanceBuffer_impl(vbo, models, instanceCount);
}

void Renderer::updateInstanceColorBuffer(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount)
{
    return sInstance->updateInstanceColorBuffer_impl(vbo, colors, instanceCount);
}

void Renderer::createSprite(unsigned int *vao)
{
    return sInstance->createSprite_impl(vao);
}

void Renderer::destroySprite(unsigned int *vao)
{
    return sInstance->destroySprite_impl(vao);
}

void Renderer::createFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int *vao,
                            unsigned int *vbo0, unsigned int *vbo1)
{
    return sInstance->createFrustum_impl(vertices, normals, vao, vbo0, vbo1);
}

void Renderer::destroyFrustum(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    return sInstance->destroyFrustum_impl(vao, vbo0, vbo1);
}

void Renderer::updateFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int vbo0,
                            unsigned int vbo1)
{
    return sInstance->updateFrustum_impl(vertices, normals, vbo0, vbo1);
}

void Renderer::updateFrustum(const std::vector<float> &vertices, unsigned int vbo0)
{
    return sInstance->updateFrustum_impl(vertices, vbo0);
}

void Renderer::createGrid(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0)
{
    return sInstance->createGrid_impl(vertices, vao, vbo0);
}

void Renderer::destroyGrid(unsigned int *vao, unsigned int *vbo0)
{
    return sInstance->destroyGrid_impl(vao, vbo0);
}

void Renderer::createLine(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                        unsigned int *vbo0, unsigned int *vbo1)
{
    return sInstance->createLine_impl(vertices, colors, vao, vbo0, vbo1);
}

void Renderer::destroyLine(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1)
{
    return sInstance->destroyLine_impl(vao, vbo0, vbo1);
}

void Renderer::preprocess(std::string &vert, std::string &frag, std::string &geom, int64_t variant)
{
    return sInstance->preprocess_impl(vert, frag, geom, variant);
}

void Renderer::compile(const std::string &name, const std::string &vert, const std::string &frag,
                    const std::string &geom, unsigned int *program, ShaderStatus &status)
{
    return sInstance->compile_impl(name, vert, frag, geom, program, status);
}

int Renderer::findUniformLocation(const char *name, int program)
{
    return sInstance->findUniformLocation_impl(name, program);
}

int Renderer::getUniformCount(int program)
{
    return sInstance->getUniformCount_impl(program);
}

int Renderer::getAttributeCount(int program)
{
    return sInstance->getAttributeCount_impl(program);
}

std::vector<ShaderUniform> Renderer::getShaderUniforms(int program)
{
    return sInstance->getShaderUniforms_impl(program);
}

std::vector<ShaderAttribute> Renderer::getShaderAttributes(int program)
{
    return sInstance->getShaderAttributes_impl(program);
}

void Renderer::setUniformBlock(const char *blockName, int bindingPoint, int program)
{
    return sInstance->setUniformBlock_impl(blockName, bindingPoint, program);
}

void Renderer::use(int program)
{
    return sInstance->use_impl(program);
}

void Renderer::unuse()
{
    return sInstance->unuse_impl();
}

void Renderer::destroy(int program)
{
    return sInstance->destroy_impl(program);
}

void Renderer::setBool(int nameLocation, bool value)
{
    return sInstance->setBool_impl(nameLocation, value);
}

void Renderer::setInt(int nameLocation, int value)
{
    return sInstance->setInt_impl(nameLocation, value);
}

void Renderer::setFloat(int nameLocation, float value)
{
    return sInstance->setFloat_impl(nameLocation, value);
}

void Renderer::setColor(int nameLocation, const Color &color)
{
    return sInstance->setColor_impl(nameLocation, color);
}

void Renderer::setColor32(int nameLocation, const Color32 &color)
{
    return sInstance->setColor32_impl(nameLocation, color);
}

void Renderer::setVec2(int nameLocation, const glm::vec2 &vec)
{
    return sInstance->setVec2_impl(nameLocation, vec);
}

void Renderer::setVec3(int nameLocation, const glm::vec3 &vec)
{
    return sInstance->setVec3_impl(nameLocation, vec);
}

void Renderer::setVec4(int nameLocation, const glm::vec4 &vec)
{
    return sInstance->setVec4_impl(nameLocation, vec);
}

void Renderer::setMat2(int nameLocation, const glm::mat2 &mat)
{
    return sInstance->setMat2_impl(nameLocation, mat);
}

void Renderer::setMat3(int nameLocation, const glm::mat3 &mat)
{
    return sInstance->setMat3_impl(nameLocation, mat);
}

void Renderer::setMat4(int nameLocation, const glm::mat4 &mat)
{
    return sInstance->setMat4_impl(nameLocation, mat);
}

void Renderer::setTexture2D(int nameLocation, int texUnit, int tex)
{
    return sInstance->setTexture2D_impl(nameLocation, texUnit, tex);
}

void Renderer::setTexture2Ds(int nameLocation, int *texUnits, int count, int *texs)
{
    return sInstance->setTexture2Ds_impl(nameLocation, texUnits, count, texs);
}

bool Renderer::getBool(int nameLocation, int program)
{
    return sInstance->getBool_impl(nameLocation, program);
}

int Renderer::getInt(int nameLocation, int program)
{
    return sInstance->getInt_impl(nameLocation, program);
}

float Renderer::getFloat(int nameLocation, int program)
{
    return sInstance->getFloat_impl(nameLocation, program);
}

Color Renderer::getColor(int nameLocation, int program)
{
    return sInstance->getColor_impl(nameLocation, program);
}

Color32 Renderer::getColor32(int nameLocation, int program)
{
    return sInstance->getColor32_impl(nameLocation, program);
}

glm::vec2 Renderer::getVec2(int nameLocation, int program)
{
    return sInstance->getVec2_impl(nameLocation, program);
}

glm::vec3 Renderer::getVec3(int nameLocation, int program)
{
    return sInstance->getVec3_impl(nameLocation, program);
}

glm::vec4 Renderer::getVec4(int nameLocation, int program)
{
    return sInstance->getVec4_impl(nameLocation, program);
}

glm::mat2 Renderer::getMat2(int nameLocation, int program)
{
    return sInstance->getMat2_impl(nameLocation, program);
}

glm::mat3 Renderer::getMat3(int nameLocation, int program)
{
    return sInstance->getMat3_impl(nameLocation, program);
}

glm::mat4 Renderer::getMat4(int nameLocation, int program)
{
    return sInstance->getMat4_impl(nameLocation, program);
}

int Renderer::getTexture2D(int nameLocation, int texUnit, int program)
{
    return sInstance->getTexture2D_impl(nameLocation, texUnit, program);
}

void Renderer::applyMaterial(const std::vector<ShaderUniform> &uniforms, int shaderProgram)
{
    return sInstance->applyMaterial_impl(uniforms, shaderProgram);
}

void Renderer::renderLines(int start, int count, int vao)
{
    return sInstance->renderLines_impl(start, count, vao);
}

void Renderer::renderLinesWithCurrentlyBoundVAO(int start, int count)
{
    return sInstance->renderLinesWithCurrentlyBoundVAO_impl(start, count);
}

void Renderer::renderWithCurrentlyBoundVAO(int start, int count)
{
    return sInstance->renderWithCurrentlyBoundVAO_impl(start, count);
}

void Renderer::render(int start, int count, int vao, bool wireframe)
{
    return sInstance->render_impl(start, count, vao, wireframe);
}

void Renderer::render(int start, int count, int vao, GraphicsQuery &query, bool wireframe)
{
    return sInstance->render_impl(start, count, vao, query, wireframe);
}

void Renderer::renderInstanced(int start, int count, int instanceCount, int vao, GraphicsQuery &query)
{
    return sInstance->renderInstanced_impl(start, count, instanceCount, vao, query);
}

void Renderer::render(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->render_impl(renderObject, query);
}

void Renderer::renderInstanced(const RenderObject &renderObject, GraphicsQuery &query)
{
    return sInstance->renderInstanced_impl(renderObject, query);
}