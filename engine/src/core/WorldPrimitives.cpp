#include "../../include/core/WorldPrimitives.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

#include <set>
#include <glm/glm.hpp>

using namespace PhysicsEngine;

void WorldPrimitives::createPrimitiveMeshes(World* world, int nx, int nz)
{
    /*Mesh *plane = world->createAsset<Mesh>(Guid("83a08619-90c8-49a0-af38-9fb3732c6bc3"));
    Mesh *disc = world->createAsset<Mesh>(Guid("a564ec96-bd6f-493a-ad2a-0009a7cb0fb0"));
    Mesh *cube = world->createAsset<Mesh>(Guid("94f5dcfc-977b-44ad-ba7c-502ce049a187"));
    Mesh *sphere = world->createAsset<Mesh>(Guid("6b415ea5-1c19-4d0d-a6df-62461167b4b3"));
    Mesh *cylinder = world->createAsset<Mesh>(Guid("af6181eb-1b8e-4102-9b6b-39ba470a87e9"));
    Mesh *cone = world->createAsset<Mesh>(Guid("d03432f9-4f06-436f-8c2f-8d3d1264da25"));*/
    Mesh *plane = world->createAsset<Mesh>();
    Mesh *disc = world->createAsset<Mesh>();
    Mesh *cube = world->createAsset<Mesh>();
    Mesh *sphere = world->createAsset<Mesh>();
    Mesh *cylinder = world->createAsset<Mesh>();
    Mesh *cone = world->createAsset<Mesh>();

    assert(plane != nullptr);
    assert(disc != nullptr);
    assert(cube != nullptr);
    assert(sphere != nullptr);
    assert(cylinder != nullptr);
    assert(cone != nullptr);

    plane->setName("Plane");
    disc->setName("Disc");
    cube->setName("Cube");
    sphere->setName("Sphere");
    cylinder->setName("Cylinder");
    cone->setName("Cone");

    /*Shader *standardShader = world->createAsset<Shader>(Guid("d875cfa8-d25b-4c5d-a26d-caafd191baf7"));*/
    Shader *standardShader = world->createAsset<Shader>();
    
    assert(standardShader != nullptr);
    
    standardShader->setName("Standard");
    standardShader->setVertexShader(Graphics::getStandardVertexShader());
    standardShader->setFragmentShader(Graphics::getStandardFragmentShader());

    std::set<ShaderMacro> variants[23];
    variants[0].insert(ShaderMacro::None);

    // directional light variants
    variants[1].insert(ShaderMacro::Directional);

    variants[2].insert(ShaderMacro::Directional);
    variants[2].insert(ShaderMacro::HardShadows);
    
    variants[3].insert(ShaderMacro::Directional);
    variants[3].insert(ShaderMacro::SoftShadows);

    variants[4].insert(ShaderMacro::Directional);
    variants[4].insert(ShaderMacro::Instancing);
    
    variants[5].insert(ShaderMacro::Directional);
    variants[5].insert(ShaderMacro::HardShadows);
    variants[5].insert(ShaderMacro::ShowCascades);

    variants[6].insert(ShaderMacro::Directional);
    variants[6].insert(ShaderMacro::HardShadows);
    variants[6].insert(ShaderMacro::ShowCascades);

    variants[7].insert(ShaderMacro::Directional);
    variants[7].insert(ShaderMacro::HardShadows);
    variants[7].insert(ShaderMacro::Instancing);

    variants[8].insert(ShaderMacro::Directional);
    variants[8].insert(ShaderMacro::HardShadows);
    variants[8].insert(ShaderMacro::Instancing);

    variants[9].insert(ShaderMacro::Directional);
    variants[9].insert(ShaderMacro::HardShadows);
    variants[9].insert(ShaderMacro::ShowCascades);
    variants[9].insert(ShaderMacro::Instancing);

    variants[10].insert(ShaderMacro::Directional);
    variants[10].insert(ShaderMacro::HardShadows);
    variants[10].insert(ShaderMacro::ShowCascades);
    variants[10].insert(ShaderMacro::Instancing);

    // spot light variants
    variants[11].insert(ShaderMacro::Spot);

    variants[12].insert(ShaderMacro::Spot);
    variants[12].insert(ShaderMacro::HardShadows);

    variants[13].insert(ShaderMacro::Spot);
    variants[13].insert(ShaderMacro::SoftShadows);

    variants[14].insert(ShaderMacro::Spot);
    variants[14].insert(ShaderMacro::Instancing);

    variants[15].insert(ShaderMacro::Spot);
    variants[15].insert(ShaderMacro::HardShadows);
    variants[15].insert(ShaderMacro::Instancing);

    variants[16].insert(ShaderMacro::Spot);
    variants[16].insert(ShaderMacro::SoftShadows);
    variants[16].insert(ShaderMacro::Instancing);

    // point light variants
    variants[17].insert(ShaderMacro::Point);

    variants[18].insert(ShaderMacro::Point);
    variants[18].insert(ShaderMacro::HardShadows);

    variants[19].insert(ShaderMacro::Point);
    variants[19].insert(ShaderMacro::SoftShadows);

    variants[20].insert(ShaderMacro::Point);
    variants[20].insert(ShaderMacro::Instancing);

    variants[21].insert(ShaderMacro::Point);
    variants[21].insert(ShaderMacro::HardShadows);
    variants[21].insert(ShaderMacro::Instancing);

    variants[22].insert(ShaderMacro::Point);
    variants[22].insert(ShaderMacro::SoftShadows);
    variants[22].insert(ShaderMacro::Instancing);

    for (int i = 0; i < 23; i++)
    {
        standardShader->addVariant(i, variants[i]);
    }

    standardShader->preprocess();
    standardShader->compile();

    /*Material *standardMaterial = world->createAsset<Material>(Guid("1d83f0b2-f16d-48e6-9cbd-20be8115179b"));*/
    Material *standardMaterial = world->createAsset<Material>();
    
    assert(standardMaterial != nullptr);

    standardMaterial->setName("Standard");
    standardMaterial->setShaderId(standardShader->getId());
    standardMaterial->onShaderChanged();

    standardMaterial->setFloat("material.shininess", 0.5f);
    standardMaterial->setVec3("material.ambient", glm::vec3(0.5f, 0.5f, 0.5f));
    standardMaterial->setVec3("material.diffuse", glm::vec3(1, 1, 1));
    standardMaterial->setVec3("material.specular", glm::vec3(0.2, 0.2, 0.2));
    standardMaterial->setVec3("material.colour", glm::vec3(0.0f, 1.0f, 1.0f));

    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .

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
            //planeVertices[i++] = xmin + x * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + z * dz;
            //
            //planeVertices[i++] = xmin + (x + 1) * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + z * dz;

            //planeVertices[i++] = xmin + x * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + (z + 1) * dz;

            //planeTexCoords[k++] = x * dx;
            //planeTexCoords[k++] = z * dz;

            //planeTexCoords[k++] = (x + 1) * dx;
            //planeTexCoords[k++] = z * dz;

            //planeTexCoords[k++] = x * dx;
            //planeTexCoords[k++] = (z + 1) * dz;

            //// second triangle
            //planeVertices[i++] = xmin + (x + 1) * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + z * dz;

            //planeVertices[i++] = xmin + (x + 1) * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + (z + 1) * dz;

            //planeVertices[i++] = xmin + x * dx;
            //planeVertices[i++] = 0.0f;
            //planeVertices[i++] = zmin + (z + 1) * dz;

            //planeTexCoords[k++] = (x + 1) * dx;
            //planeTexCoords[k++] = z * dz;

            //planeTexCoords[k++] = (x + 1) * dx;
            //planeTexCoords[k++] = (z + 1) * dz;

            //planeTexCoords[k++] = x * dx;
            //planeTexCoords[k++] = (z + 1) * dz;

            for (int n = 0; n < 6; n++)
            {
                planeNormals[j++] = 0.0f;
                planeNormals[j++] = 1.0f;
                planeNormals[j++] = 0.0f;
            }
        }
    }

    plane->load(planeVertices, planeNormals, planeTexCoords, {0, 3 * vertexCount});

    // Generate disc
    int discTriangleCount = nx * nz;
    int discVertexCount = 3 * discTriangleCount;

    std::vector<float> discVertices(3 * discVertexCount);
    std::vector<float> discNormals(3 * discVertexCount);
    std::vector<float> discTexCoords(2 * discVertexCount);

    float dtheta = 360.0f / discTriangleCount;

    i = 0;
    j = 0;
    k = 0;
    for (int s = 0; s < discTriangleCount; s++)
    {
        float theta0 = glm::radians(dtheta * s);
        float theta1 = glm::radians(dtheta * (s + 1));

        discVertices[i++] = 0.5f * glm::cos(theta0);
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.5f * glm::sin(theta0);

        discVertices[i++] = 0.5f * glm::cos(theta1);
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.5f * glm::sin(theta1);

        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.0f;

        for (int n = 0; n < 3; n++)
        {
            discNormals[j++] = 0.0f;
            discNormals[j++] = 1.0f;
            discNormals[j++] = 0.0f;
        }

        discTexCoords[k++] = 0.5f * glm::cos(theta0) + 0.5f;
        discTexCoords[k++] = 0.5f * glm::sin(theta0) + 0.5f;

        discTexCoords[k++] = 0.5f * glm::cos(theta1) + 0.5f;
        discTexCoords[k++] = 0.5f * glm::sin(theta1) + 0.5f;

        discTexCoords[k++] = 0.5f;
        discTexCoords[k++] = 0.5f;
    }

    disc->load(discVertices, discNormals, discTexCoords, {0, 3 * discVertexCount});

    // Generate cube mesh
    int cubeVertexCount = 6 * vertexCount;

    std::vector<float> cubeVertices(3 * cubeVertexCount);
    std::vector<float> cubeNormals(3 * cubeVertexCount);
    std::vector<float> cubeTexCoords(2 * cubeVertexCount);
    
    int faceStart = 0;
    int texStart = 0;

    // bottom face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 1 + faceStart] = -0.5f;
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 0];

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = -1.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];

    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // top face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 1 + faceStart] = 0.5f;
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 1.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // left face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = -0.5f;
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = -1.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // right face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = 0.5f;
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 0];

        cubeNormals[3 * v + 0 + faceStart] = 1.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // near face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 2 + faceStart] = -0.5;

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = -1.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // far face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = 0.5f;

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 1.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    cube->load(cubeVertices, cubeNormals, cubeTexCoords, {0, 3 * cubeVertexCount});

    // Generate Sphere mesh
    int sphereVertexCount = cubeVertexCount;

    std::vector<float> sphereVertices(3 * sphereVertexCount);
    std::vector<float> sphereNormals(3 * sphereVertexCount);
    std::vector<float> sphereTexCoords(2 * sphereVertexCount);

    for (int v = 0; v < sphereVertexCount; v++)
    {
        float vx = cubeVertices[3 * v + 0];
        float vy = cubeVertices[3 * v + 1];
        float vz = cubeVertices[3 * v + 2];

        float length = glm::sqrt(vx * vx + vy * vy + vz * vz);

        vx = vx / length;
        vy = vy / length;
        vz = vz / length;
        
        sphereVertices[3 * v + 0] = vx;
        sphereVertices[3 * v + 1] = vy;
        sphereVertices[3 * v + 2] = vz;
            
        sphereNormals[3 * v + 0] = vx;
        sphereNormals[3 * v + 1] = vy;
        sphereNormals[3 * v + 2] = vz;

        sphereTexCoords[2 * v + 0] = cubeTexCoords[2 * v + 0];
        sphereTexCoords[2 * v + 1] = cubeTexCoords[2 * v + 1];
    }

    sphere->load(sphereVertices, sphereNormals, sphereTexCoords, {0, 3 * sphereVertexCount});

    // Generate cylinder
    int cylinderTriangleCount = 4 * discTriangleCount;
    int cylinderVertexCount = 3 * cylinderTriangleCount;
    
    std::vector<float> cylinderVertices(3 * cylinderVertexCount);
    std::vector<float> cylinderNormals(3 * cylinderVertexCount);
    std::vector<float> cylinderTexCoords(3 * cylinderVertexCount);

    faceStart = 0;
    texStart = 0;

    // bottom
    for (int v = 0; v < discVertexCount; v++)
    {
        cylinderVertices[3 * v + 0 + faceStart] = discVertices[3 * v + 0];
        cylinderVertices[3 * v + 1 + faceStart] = -0.5f;
        cylinderVertices[3 * v + 2 + faceStart] = discVertices[3 * v + 2];

        cylinderNormals[3 * v + 0 + faceStart] = 0.0f;
        cylinderNormals[3 * v + 1 + faceStart] = -1.0f;
        cylinderNormals[3 * v + 2 + faceStart] = 0.0f;

        cylinderTexCoords[2 * v + 0 + texStart] = discTexCoords[2 * v + 0];
        cylinderTexCoords[2 * v + 1 + texStart] = discTexCoords[2 * v + 1];
    }

    faceStart += 3 * discVertexCount;
    texStart += 2 * discVertexCount;

    // top
    for (int v = 0; v < discVertexCount; v++)
    {
        cylinderVertices[3 * v + 0 + faceStart] = discVertices[3 * v + 0];
        cylinderVertices[3 * v + 1 + faceStart] = 0.5f;
        cylinderVertices[3 * v + 2 + faceStart] = discVertices[3 * v + 2];

        cylinderNormals[3 * v + 0 + faceStart] = 0.0f;
        cylinderNormals[3 * v + 1 + faceStart] = 1.0f;
        cylinderNormals[3 * v + 2 + faceStart] = 0.0f;

        cylinderTexCoords[2 * v + 0 + texStart] = discTexCoords[2 * v + 0];
        cylinderTexCoords[2 * v + 1 + texStart] = discTexCoords[2 * v + 1];
    }

    faceStart += 3 * discVertexCount;
    texStart += 2 * discVertexCount;

    // tube
    i = faceStart;
    j = faceStart;
    k = texStart;
    /*for (int v = 0; v < discVertexCount; v++)
    {
        // skip center disc vertex
        if ((v + 1) % 3 != 0)
        {
            // first triangle
            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            // second triangle
            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];
        }
    }*/

    cylinder->load(cylinderVertices, cylinderNormals, cylinderTexCoords, {0, 3 * cylinderVertexCount});

    mPlaneMeshId = plane->getId();
    mDiscMeshId = disc->getId();
    mCubeMeshId = cube->getId();
    mSphereMeshId = sphere->getId();
    mCylinderMeshId = cylinder->getId();
    mConeMeshId = cone->getId();

    mStandardShaderId = standardShader->getId();
    mStandardMaterialId = standardMaterial->getId();
}