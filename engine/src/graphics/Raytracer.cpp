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

void Raytracer::update(Camera *camera, const TLAS &tlas, const std::vector<BLAS *> &blas,
                       const std::vector<glm::mat4> &models, const BVH &bvh, const std::vector<Sphere> &spheres)
{
    // Spheres
    /*std::vector<RaytraceMaterial> materials(spheres.size());
    materials[0].mType = RaytraceMaterial::MaterialType::Lambertian;
    materials[0].mAlbedo = glm::vec3(0.8f, 0.8f, 0.8f);
    materials[1].mType = RaytraceMaterial::MaterialType::Lambertian;
    materials[1].mAlbedo = glm::vec3(0.8f, 0.4f, 0.4f);
    materials[2].mType = RaytraceMaterial::MaterialType::Lambertian;
    materials[2].mAlbedo = glm::vec3(0.4f, 0.8f, 0.4f);

    materials[3].mType = RaytraceMaterial::MaterialType::Metallic;
    materials[3].mAlbedo = glm::vec3(0.4f, 0.4f, 0.8f);
    materials[3].mFuzz = 0.0f;
    materials[4].mType = RaytraceMaterial::MaterialType::Metallic;
    materials[4].mAlbedo = glm::vec3(0.4f, 0.8f, 0.4f);
    materials[4].mFuzz = 0.0f;
    materials[5].mType = RaytraceMaterial::MaterialType::Metallic;
    materials[5].mAlbedo = glm::vec3(0.4f, 0.8f, 0.4f);
    materials[5].mFuzz = 0.2f;
    materials[6].mType = RaytraceMaterial::MaterialType::Metallic;
    materials[6].mAlbedo = glm::vec3(0.4f, 0.8f, 0.4f);
    materials[6].mFuzz = 0.6f;

    materials[7].mType = RaytraceMaterial::MaterialType::Dialectric;
    materials[7].mAlbedo = glm::vec3(0.4f, 0.4f, 0.4f);
    materials[7].mRefractionIndex = 1.5f;
    materials[8].mType = RaytraceMaterial::MaterialType::DiffuseLight;
    materials[8].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
    materials[8].mEmissive = glm::vec3(30.0f, 25.0f, 15.0f);

    if (camera->moved())
    {
        camera->clearPixels();
        //camera->clearPixelsUsingDevice();
    }
    
    camera->resizePixels();
    //camera->resizePixelsUsingDevice();

    auto start = std::chrono::high_resolution_clock::now();
    camera->raytraceSpheres(bvh, spheres, materials, 5, 32);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;
    std::cout << "Based MRays/s: " << ((256 * 256) / elapsed_time.count()) / 1000000.0f << "\n";

    camera->updateFinalImage();
    //camera->updateFinalImageUsingDevice();*/






    //// TLAS and BLAS
    //std::vector<RaytraceMaterial> materials(5);
    //materials[0].mType = RaytraceMaterial::MaterialType::Lambertian;
    //materials[0].mAlbedo = glm::vec3(0.1f, 0.2f, 0.5f);

    //materials[1].mType = RaytraceMaterial::MaterialType::Dialectric;
    //materials[1].mAlbedo = glm::vec3(1.0f, 1.0f, 1.0f);
    //materials[1].mRefractionIndex = 1.5f;

    //materials[2].mType = RaytraceMaterial::MaterialType::Lambertian;
    //materials[2].mAlbedo = glm::vec3(0.8f, 0.2f, 0.5f);

    //materials[3].mType = RaytraceMaterial::MaterialType::Metallic;
    //materials[3].mAlbedo = glm::vec3(0.8f, 0.6f, 0.2f);
    //materials[3].mFuzz = 0.0f;

    //materials[4].mType = RaytraceMaterial::MaterialType::DiffuseLight;
    //materials[4].mEmissive = glm::vec3(4.0f, 4.0f, 4.0f);

    //if (camera->moved())
    //{
    //    camera->clearPixels();
    //}

    //camera->resizePixels();

    //auto start = std::chrono::high_resolution_clock::now();
    //camera->raytraceScene(tlas, blas, models, materials, 5, 32);
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed_time = end - start;

    //camera->updateFinalImage();

    //std::cout << "Based MRays/s: " << ((256 * 256) / elapsed_time.count()) / 1000000.0f << "\n";


















    size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();
    std::vector<RaytraceMaterial> materials(meshRendererCount);
    for (size_t i = 0; i < meshRendererCount; i++)
    {
        materials[i].mType = (i % 2 == 0) ? RaytraceMaterial::MaterialType::Lambertian : RaytraceMaterial::MaterialType::Metallic;
        if (i == 3)
        {
            materials[i].mType = RaytraceMaterial::MaterialType::DiffuseLight;
            materials[i].mEmissive = glm::vec3(30.0f, 25.0f, 15.0f);
        }
        
        materials[i].mAlbedo = (i % 2 == 0) ? glm::vec3(0.8f, 0.2f, 0.5f) : glm::vec3(0.8f, 0.6f, 0.2f);
        materials[i].mFuzz = 0.0f;
    }

    if (camera->moved())
    {
        camera->clearPixels();
    }

    camera->resizePixels();

    auto start = std::chrono::high_resolution_clock::now();
    camera->raytraceScene(tlas, blas, models, materials, 5, 32);
    //camera->raytraceNormals(tlas, blas, models, 32);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    camera->updateFinalImage();

    std::cout << "Based MRays/s: " << ((256 * 256) / elapsed_time.count()) / 1000000.0f << "\n";
}