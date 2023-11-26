#include "../../include/core/Log.h"

#include "../../include/graphics/Renderer.h"
#include "../../include/graphics/RendererMeshes.h"

using namespace PhysicsEngine;

// Meshes generated using clockwise triangle winding order. Note: For OpenGL, the default
// windng order is counter-clockwise (CCW) while for DirectX the default is clockwise (CW)

ScreenQuad::ScreenQuad()
{
    mMesh = MeshHandle::create();

    mVertexBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();

    // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
    // clang-format off
    float quadVertices[] = {
        // positions
        -1.0f, 1.0f, 
        -1.0f, -1.0f, 
        1.0f, -1.0f,
        -1.0f, 1.0f,
        1.0f,  -1.0f, 
        1.0f, 1.0f};
    // clang-format on

    // clang-format off
    float quadTexCoords[] = {
        // texCoords
        0.0f, 1.0f, 
        0.0f, 0.0f, 
        1.0f, 0.0f,
        0.0f, 1.0f, 
        1.0f, 0.0f, 
        1.0f, 1.0f};
    // clang-format on

    mVertexBuffer->bind(0);
    mVertexBuffer->resize(sizeof(float) * 12);
    mVertexBuffer->setData(quadVertices, 0, sizeof(float) * 12);
    mVertexBuffer->unbind(0);

    mTexCoordsBuffer->bind(1);
    mTexCoordsBuffer->resize(sizeof(float) * 12);
    mTexCoordsBuffer->setData(quadTexCoords, 0, sizeof(float) * 12);
    mTexCoordsBuffer->unbind(1);

    mMesh->addVertexBuffer(mVertexBuffer, "POSITION", AttribType::Vec2);
    mMesh->addVertexBuffer(mTexCoordsBuffer, "TEXCOORD", AttribType::Vec2);
}

ScreenQuad::~ScreenQuad()
{
    delete mVertexBuffer;
    delete mTexCoordsBuffer;

    delete mMesh;
}

void ScreenQuad::bind()
{
    mMesh->bind();
}

void ScreenQuad::unbind()
{
    mMesh->unbind();
}

void ScreenQuad::draw()
{
    Renderer::turnOff(Capability::Depth_Testing);
    mMesh->draw(0, 6);
    Renderer::turnOn(Capability::Depth_Testing);
}

ScreenQuad *RendererMeshes::sScreenQuad = nullptr;

void RendererMeshes::createInternalMeshes()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    Log::warn("Start building internal meshes\n");
    sScreenQuad = new ScreenQuad();
    Log::warn("Finished building internal meshes\n");
}

ScreenQuad *RendererMeshes::getScreenQuad()
{
    return RendererMeshes::sScreenQuad;
}