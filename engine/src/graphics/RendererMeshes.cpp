#include "../../include/core/Log.h"

#include "../../include/graphics/RendererMeshes.h"

using namespace PhysicsEngine;

// Meshes generated using clockwise triangle winding order. Note: For OpenGL, the default 
// windng order is counter-clockwise (CCW) while for DirectX the default is clockwise (CW)

PlaneMesh::PlaneMesh(int nx, int nz)
{
    // Generate plane mesh
    int triangleCount = 2 * (nx - 1) * (nz - 1);
    int vertexCount = 3 * triangleCount;

    std::vector<float> planeVertices(3 * vertexCount);
    std::vector<float> planeNormals(3 * vertexCount);
    std::vector<float> planeTexCoords(2 * vertexCount);

    float xmin = -0.5f;
    float xmax = 0.5f;
    float zmin = -0.5f;
    float zmax = 0.5f;

    float dx = (xmax - xmin) / (nx - 1);
    float dz = (zmax - zmin) / (nz - 1);

    int i = 0;
    int j = 0;
    int k = 0;
    for (int z = 0; z < (nz - 1); z++)
    {
        for (int x = 0; x < (nx - 1); x++)
        {
            // first triangle (clockwise winding order)
            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;

            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = z * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = z * dz;

            // second triangle (clockwise winding order)
            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;

            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = z * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            //// first triangle
            // planeVertices[i++] = xmin + x * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + z * dz;
            //
            // planeVertices[i++] = xmin + (x + 1) * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + z * dz;

            // planeVertices[i++] = xmin + x * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + (z + 1) * dz;

            // planeTexCoords[k++] = x * dx;
            // planeTexCoords[k++] = z * dz;

            // planeTexCoords[k++] = (x + 1) * dx;
            // planeTexCoords[k++] = z * dz;

            // planeTexCoords[k++] = x * dx;
            // planeTexCoords[k++] = (z + 1) * dz;

            //// second triangle
            // planeVertices[i++] = xmin + (x + 1) * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + z * dz;

            // planeVertices[i++] = xmin + (x + 1) * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + (z + 1) * dz;

            // planeVertices[i++] = xmin + x * dx;
            // planeVertices[i++] = 0.0f;
            // planeVertices[i++] = zmin + (z + 1) * dz;

            // planeTexCoords[k++] = (x + 1) * dx;
            // planeTexCoords[k++] = z * dz;

            // planeTexCoords[k++] = (x + 1) * dx;
            // planeTexCoords[k++] = (z + 1) * dz;

            // planeTexCoords[k++] = x * dx;
            // planeTexCoords[k++] = (z + 1) * dz;

            for (int n = 0; n < 6; n++)
            {
                planeNormals[j++] = 0.0f;
                planeNormals[j++] = 1.0f;
                planeNormals[j++] = 0.0f;
            }
        }
    }

    std::vector<unsigned int> planeIndices(vertexCount);
    for (int ii = 0; ii < vertexCount; ii++)
    {
        planeIndices[ii] = ii;
    }
    //plane->load(planeVertices, planeNormals, planeTexCoords, planeIndices, {0, vertexCount});


    mVertexBuffer = VertexBuffer::create();
    mNormalBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();
    mInstanceModelBuffer = VertexBuffer::create();
    mInstanceColorBuffer = VertexBuffer::create();
    mIndexBuffer = IndexBuffer::create();
    mMesh = MeshHandle::create();

    mMesh->addVertexBuffer(mVertexBuffer, AttribType::Vec3);
    mMesh->addVertexBuffer(mNormalBuffer, AttribType::Vec3);
    mMesh->addVertexBuffer(mTexCoordsBuffer, AttribType::Vec2);
    mMesh->addVertexBuffer(mInstanceModelBuffer, AttribType::Mat4, true);
    mMesh->addVertexBuffer(mInstanceColorBuffer, AttribType::UVec4, true);
    mMesh->addIndexBuffer(mIndexBuffer);
   
    mVertexBuffer->bind();
    mVertexBuffer->resize(sizeof(float) * mVertices.size());
    mVertexBuffer->setData(mVertices.data(), 0, sizeof(float) * mVertices.size());
    mVertexBuffer->unbind();

    mNormalBuffer->bind();
    mNormalBuffer->resize(sizeof(float) * mNormals.size());
    mNormalBuffer->setData(mNormals.data(), 0, sizeof(float) * mNormals.size());
    mNormalBuffer->unbind();

    mTexCoordsBuffer->bind();
    mTexCoordsBuffer->resize(sizeof(float) * mTexCoords.size());
    mTexCoordsBuffer->setData(mTexCoords.data(), 0, sizeof(float) * mTexCoords.size());
    mTexCoordsBuffer->unbind();

    mInstanceModelBuffer->bind();
    mInstanceModelBuffer->resize(sizeof(glm::mat4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    // mInstanceModelBuffer->setData(nullptr, 0, sizeof(glm::mat4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    mInstanceModelBuffer->unbind();

    mInstanceColorBuffer->bind();
    mInstanceColorBuffer->resize(sizeof(glm::uvec4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    // mInstanceColorBuffer->setData(nullptr, 0, sizeof(glm::uvec4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    mInstanceColorBuffer->unbind();

    mIndexBuffer->bind();
    mIndexBuffer->resize(sizeof(unsigned int) * mIndices.size());
    mIndexBuffer->setData(mIndices.data(), 0, sizeof(unsigned int) * mIndices.size());
    mIndexBuffer->unbind();
}

PlaneMesh::~PlaneMesh()
{
    delete mMesh;
}

void PlaneMesh::bind()
{
    mMesh->bind();
}

void PlaneMesh::unbind()
{
    mMesh->unbind();
}

SphereMesh::SphereMesh(float radius) : mRadius(radius)
{
    mMesh = MeshHandle::create();

}

SphereMesh::~SphereMesh()
{
    delete mMesh;
}

void SphereMesh::bind()
{
    mMesh->bind();
}

void SphereMesh::unbind()
{
    mMesh->unbind();
}

SphereMesh *RendererMeshes::sSphereMesh = nullptr;

void RendererMeshes::createInternalMeshes()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    Log::warn("Start building internal meshes\n");
    sSphereMesh = new SphereMesh();
    Log::warn("Finished building internal meshes\n");
}

SphereMesh *RendererMeshes::getSphereMesh()
{
    return RendererMeshes::sSphereMesh;
}