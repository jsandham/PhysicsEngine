#include <vector>
#include <random>
#include <chrono>

#include "../../include/graphics/Raytracer.h"

#include "../../include/core/World.h"

using namespace PhysicsEngine;

Raytracer::Raytracer()
{
}

Raytracer ::~Raytracer()
{

}

void Raytracer::init(World *world)
{
    mWorld = world;
}

void Raytracer::update(Camera *camera)
{
    //// Spheres
    //srand(0);
    //int sphereCount = 100;
    //std::vector<Sphere> spheres(sphereCount);
    //spheres[0] = Sphere(glm::vec3(0.0, -100.5, -1.0f), 100.0f);
    //spheres[1] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), 0.5f);
    //spheres[2] = Sphere(glm::vec3(-1.0, 0.0, -1.0f), -0.4f);
    //spheres[3] = Sphere(glm::vec3(0.0, 0.0, -1.0f), 0.5f);
    //spheres[4] = Sphere(glm::vec3(1.0, 0.0, -1.0f), 0.5f);
    //spheres[5] = Sphere(glm::vec3(0.75f, 2.25f, -0.5f), 0.7f);
    //for (int i = 6; i < sphereCount; i++)
    //{
    //    spheres[i] = Sphere(glm::linearRand(glm::vec3(-20.0f, 0.0f, -20.0f), glm::vec3(20.0f, 0.0f, 20.0f)), glm::linearRand(0.4f, 2.0f));
    //}

    //std::vector<RaytraceMaterial> materials(sphereCount);
    //materials[0].mAlbedo = glm::vec3(0.8f, 0.8f, 0.0f);

    //materials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
    //materials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
    //materials[1].mRefractionIndex = 1.5f;

    //materials[2].mType = RaytraceMaterial::MaterialType::Dialectric;
    //materials[2].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
    //materials[2].mRefractionIndex = 1.5f;

    //materials[3].mType = RaytraceMaterial::MaterialType::Lambertian;
    //materials[3].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

    //materials[4].mType = RaytraceMaterial::MaterialType::Metallic;
    //materials[4].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
    //materials[4].mFuzz = 0.1f;

    //materials[5].mType = RaytraceMaterial::MaterialType::Lambertian;
    //materials[5].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

    //for (int i = 6; i < sphereCount; i++)
    //{
    //    materials[i].mType =
    //        (i % 2 == 0) ? RaytraceMaterial::MaterialType::Lambertian : RaytraceMaterial::MaterialType::Metallic;
    //    materials[i].mAlbedo = glm::linearRand(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
    //    materials[i].mFuzz = glm::linearRand(0.0f, 0.2f);
    //}

    //std::vector<AABB> boundingVolumes(sphereCount);
    //for (int i = 0; i < sphereCount; i++)
    //{
    //    boundingVolumes[i].mCentre = spheres[i].mCentre;
    //    boundingVolumes[i].mSize = 2.0f * glm::vec3(spheres[i].mRadius, spheres[i].mRadius, spheres[i].mRadius);
    //}

    //BVH bvh;
    //bvh.allocateBVH(boundingVolumes.size());
    //bvh.buildBVH(boundingVolumes.data(), boundingVolumes.size());
    //if (camera->moved())
    //{
    //    camera->clearPixels();
    //}
    //
    //camera->resizePixels();
    //
    //camera->raytraceSpheres(bvh, spheres, materials, 5, 32);
    //
    //camera->updateFinalImage();
    //bvh.freeBVH();




    // TLAS and BLAS
    static bool generate_blas = true;
    if (generate_blas)
    {
        Mesh *planeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Plane);
        Mesh *sphereMesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);
        Mesh *cubeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Cube);

        std::vector<float> planeVertices = planeMesh->getVertices();
        std::vector<float> sphereVertices = sphereMesh->getVertices();
        std::vector<float> cubeVertices = cubeMesh->getVertices();

        std::vector<unsigned int> planeIndices = planeMesh->getIndices();
        std::vector<unsigned int> sphereIndices = sphereMesh->getIndices();
        std::vector<unsigned int> cubeIndices = cubeMesh->getIndices();

        std::vector<Triangle> planeTriangles(planeIndices.size() / 3);
        std::vector<Triangle> sphereTriangles(sphereIndices.size() / 3);
        std::vector<Triangle> cubeTriangles(cubeIndices.size() / 3);

        for (size_t i = 0; i < planeIndices.size() / 3; i++)
        {
            unsigned int i0 = planeIndices[3 * i + 0];
            unsigned int i1 = planeIndices[3 * i + 1];
            unsigned int i2 = planeIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(planeVertices[3 * i0 + 0], planeVertices[3 * i0 + 1], planeVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(planeVertices[3 * i1 + 0], planeVertices[3 * i1 + 1], planeVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(planeVertices[3 * i2 + 0], planeVertices[3 * i2 + 1], planeVertices[3 * i2 + 2]);

            planeTriangles[i].mV0 = v0;
            planeTriangles[i].mV1 = v1;
            planeTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < sphereIndices.size() / 3; i++)
        {
            unsigned int i0 = sphereIndices[3 * i + 0];
            unsigned int i1 = sphereIndices[3 * i + 1];
            unsigned int i2 = sphereIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(sphereVertices[3 * i0 + 0], sphereVertices[3 * i0 + 1], sphereVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(sphereVertices[3 * i1 + 0], sphereVertices[3 * i1 + 1], sphereVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(sphereVertices[3 * i2 + 0], sphereVertices[3 * i2 + 1], sphereVertices[3 * i2 + 2]);

            sphereTriangles[i].mV0 = v0;
            sphereTriangles[i].mV1 = v1;
            sphereTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < cubeIndices.size() / 3; i++)
        {
            unsigned int i0 = cubeIndices[3 * i + 0];
            unsigned int i1 = cubeIndices[3 * i + 1];
            unsigned int i2 = cubeIndices[3 * i + 2];

            glm::vec3 v0 =
                glm::vec3(cubeVertices[3 * i0 + 0], cubeVertices[3 * i0 + 1], cubeVertices[3 * i0 + 2]);
            glm::vec3 v1 =
                glm::vec3(cubeVertices[3 * i1 + 0], cubeVertices[3 * i1 + 1], cubeVertices[3 * i1 + 2]);
            glm::vec3 v2 =
                glm::vec3(cubeVertices[3 * i2 + 0], cubeVertices[3 * i2 + 1], cubeVertices[3 * i2 + 2]);

            cubeTriangles[i].mV0 = v0;
            cubeTriangles[i].mV1 = v1;
            cubeTriangles[i].mV2 = v2;
        }

        glm::mat4 planeModel = glm::mat4(1.0f);
        planeModel[0] *= 10.0f;
        planeModel[1] *= 10.0f;
        planeModel[2] *= 10.0f;
        glm::mat4 sphereModelLeft = glm::mat4(1.0f);
        sphereModelLeft[3] = glm::vec4(-2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelCentre = glm::mat4(1.0f);
        sphereModelCentre[3] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelRight = glm::mat4(1.0f);
        sphereModelRight[3] = glm::vec4(2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 cubeModel = glm::mat4(1.0f);
        cubeModel[3] = glm::vec4(0.0f, 3.0f, 0.0f, 1.0f);

        mBLAS.resize(5);
        mBLAS[0].allocateBLAS(planeTriangles.size());
        mBLAS[0].buildBLAS(planeTriangles, planeModel, planeTriangles.size());
        mBLAS[1].allocateBLAS(sphereTriangles.size());
        mBLAS[1].buildBLAS(sphereTriangles, sphereModelLeft, sphereTriangles.size());
        mBLAS[2].allocateBLAS(sphereTriangles.size());
        mBLAS[2].buildBLAS(sphereTriangles, sphereModelCentre, sphereTriangles.size());
        mBLAS[3].allocateBLAS(sphereTriangles.size());
        mBLAS[3].buildBLAS(sphereTriangles, sphereModelRight, sphereTriangles.size());
        mBLAS[4].allocateBLAS(cubeTriangles.size());
        mBLAS[4].buildBLAS(cubeTriangles, cubeModel, cubeTriangles.size());

        mMaterials.resize(5);
        mMaterials[0].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[0].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

        mMaterials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
        mMaterials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
        mMaterials[1].mRefractionIndex = 1.5f;

        mMaterials[2].mType = RaytraceMaterial::MaterialType::Lambertian;
        mMaterials[2].mAlbedo = glm::vec3(0.8f, 0.2f, 0.5f);

        mMaterials[3].mType = RaytraceMaterial::MaterialType::Metallic;
        mMaterials[3].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
        mMaterials[3].mFuzz = 0.0f;

        mMaterials[4].mType = RaytraceMaterial::MaterialType::DiffuseLight;
        mMaterials[4].mEmissive = glm::vec3(4.0f, 4.0f, 4.0f);

        generate_blas = false;
    }


    mTLAS.allocateTLAS(5);
    mTLAS.buildTLAS(mBLAS.data(), 5);

    if (camera->moved())
    {
        camera->clearPixels();
    }

    camera->resizePixels();

    auto start = std::chrono::high_resolution_clock::now();
    camera->raytraceScene(mTLAS, mBLAS, mMaterials, 5, 32);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    camera->updateFinalImage();

    mTLAS.freeTLAS();

    std::cout << "Based MRays/s: " << ((256 * 256) / elapsed_time.count()) / 1000000.0f << "\n";
}