
//#define __HIP_PLATFORM_NVIDIA__
//#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../../../../../include/graphics/platform/gpgpu/gpgpu.h"

using namespace gpgpu;

#define CHECK_CUDA(ans)                          \
{                                                \
    gpuAssert((ans), __FILE__, __LINE__);        \
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

struct vec2
{
    union {
        struct
        {
            float x, y;
        };
        struct
        {
            float r, g;
        };
    };

    __device__ vec2() : x(0.0f), y(0.0f){};
    __device__ vec2(float x, float y) : x(x), y(y){};

    __device__ vec2 operator+=(const vec2 &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }

    __device__ vec2 operator-=(const vec2 &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }

    __device__ vec2 operator*=(float scalar)
    {
        this->x *= scalar;
        this->y *= scalar;
        return *this;
    }

    __device__ vec2 operator*=(const vec2 &vec)
    {
        this->x *= vec.x;
        this->y *= vec.y;
        return *this;
    }

    __device__ vec2 operator/=(const vec2 &vec)
    {
        this->x /= vec.x;
        this->y /= vec.y;
        return *this;
    }
};

__device__ vec2 operator*(const vec2 &v, float scalar)
{
    return vec2(v.x * scalar, v.y * scalar);
}

__device__ vec2 operator+(const vec2 &v1, const vec2 &v2)
{
    return vec2(v1.x + v2.x, v1.y + v2.y);
}

struct vec3
{
    union {
        struct
        {
            float x, y, z;
        };
        struct
        {
            float r, g, b;
        };
    };

    __device__ vec3() : x(0.0f), y(0.0f), z(0.0f){};
    __device__ vec3(float x, float y, float z) : x(x), y(y), z(z){};

    __device__ vec3 operator+=(const vec3 &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
        return *this;
    }

    __device__ vec3 operator-=(const vec3 &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        this->z -= vec.z;
        return *this;
    }

    __device__ vec3 operator*=(float scalar)
    {
        this->x *= scalar;
        this->y *= scalar;
        this->z *= scalar;
        return *this;
    }

    __device__ vec3 operator*=(const vec3 &vec)
    {
        this->x *= vec.x;
        this->y *= vec.y;
        this->z *= vec.z;
        return *this;
    }

    __device__ vec3 operator/=(const vec3 &vec)
    {
        this->x /= vec.x;
        this->y /= vec.y;
        this->z /= vec.z;
        return *this;
    }
};

__device__ vec3 operator*(const vec3 &v, float scalar)
{
    return vec3(v.x * scalar, v.y * scalar, v.z * scalar);
}

__device__ vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
__device__ float clamp(float d, float min, float max)
{
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

__device__ vec3 mix(vec3 x, vec3 y, float alpha)
{
    return x * (1.0f - alpha) + y * alpha;
}

__device__ uint32_t pcg_hash(uint32_t seed)
{
    uint32_t state = seed * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ float generate_rand(float a = 0.0f, float b = 1.0f)
{
    static uint32_t seed = 1234567;
    seed++;

    float uniform = (float)pcg_hash(seed) / (float)UINT32_MAX;
    return a + (b - a) * uniform;
}

struct Ray
{
    vec3 mOrigin;
    vec3 mDirection;
};
template <unsigned int BLOCKSIZE, typename T> __global__ void set_array_to_value_kernel(T *data, int size, T val)
{
    int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

    if (tid < size)
    {
        data[tid] = val;
    }
}

template <unsigned int BLOCKSIZE> __global__ 
void update_final_image_kernel(const float *image, 
                               const int *samplesPerPixel,
                               const int *intersectionCount,
                               unsigned char *finalImage,
                               unsigned char *finalIntersectionCountImage,
                               int width, 
                               int height)
{
    int tid = BLOCKSIZE * blockIdx.x + threadIdx.x;

    if (tid < width * height)
    {
        int sampleCount = samplesPerPixel[tid];

        // Read color from image
        float r = image[3 * tid + 0];
        float g = image[3 * tid + 1];
        float b = image[3 * tid + 2];

        float scale = 1.0f / sampleCount;
        r *= scale;
        g *= scale;
        b *= scale;

        // Gamma correction
        r = sqrt(r);
        g = sqrt(g);
        b = sqrt(b);

        int ir = (int)(255 * clamp(r, 0.0f, 1.0f));
        int ig = (int)(255 * clamp(g, 0.0f, 1.0f));
        int ib = (int)(255 * clamp(b, 0.0f, 1.0f));

        finalImage[3 * tid + 0] = (sampleCount > 0) ? static_cast<unsigned char>(ir) : 0;
        finalImage[3 * tid + 1] = (sampleCount > 0) ? static_cast<unsigned char>(ig) : 0;
        finalImage[3 * tid + 2] = (sampleCount > 0) ? static_cast<unsigned char>(ib) : 0;

        // Debug intersection visualization
        int count = (sampleCount > 0) ? intersectionCount[tid] / sampleCount : 0;

        float alpha = count / 70.0f;
        vec3 c;
        if (alpha < 0.5f)
        {
            c = mix(vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f), alpha);
        }
        else
        {
            c = mix(vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), alpha);
        }

        int ir2 = (int)(255 * clamp(c.r, 0.0f, 1.0f));
        int ig2 = (int)(255 * clamp(c.g, 0.0f, 1.0f));
        int ib2 = (int)(255 * clamp(c.b, 0.0f, 1.0f));

        finalIntersectionCountImage[3 * tid + 0] = static_cast<unsigned char>(ir2);
        finalIntersectionCountImage[3 * tid + 1] = static_cast<unsigned char>(ig2);
        finalIntersectionCountImage[3 * tid + 2] = static_cast<unsigned char>(ib2);
    }
}

void gpgpu::clearPixels(float *image, int *samplesPerPixel, int* intersectionCount, int width, int height)
{
    int image_size = 3 * width * height;
    int samples_size = width * height;

    int *d_samplesPerPixel = nullptr;
    int *d_intersectionCount = nullptr;
    float *d_image = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_samplesPerPixel, sizeof(int) * samples_size));
    CHECK_CUDA(cudaMalloc((void **)&d_intersectionCount, sizeof(int) * samples_size));
    CHECK_CUDA(cudaMalloc((void **)&d_image, sizeof(float) * image_size));

    set_array_to_value_kernel<256>
        <<<dim3((samples_size - 1) / 256 + 1), dim3(256), 0, 0>>>(d_samplesPerPixel, samples_size, 0);
    set_array_to_value_kernel<256>
        <<<dim3((samples_size - 1) / 256 + 1), dim3(256), 0, 0>>>(d_intersectionCount, samples_size, 0);
    set_array_to_value_kernel<256><<<dim3((image_size - 1) / 256 + 1), dim3(256), 0, 0>>>(d_image, image_size, 0.0f);

    CHECK_CUDA(cudaMemcpy(samplesPerPixel, d_samplesPerPixel, sizeof(int) * samples_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(intersectionCount, d_intersectionCount, sizeof(int) * samples_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(image, d_image, sizeof(float) * image_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_samplesPerPixel));
    CHECK_CUDA(cudaFree(d_intersectionCount));
    CHECK_CUDA(cudaFree(d_image));
}

void gpgpu::raytraceNormals(float *image, glm::vec3 cameraPosition, glm::mat4 projectionMatrix, int width, int height)
{

}

void gpgpu::updateFinalImage(const float *image, const int *samplesPerPixel, const int *intersectionCount, unsigned char *finalImage, unsigned char *finalIntersectionCountImage, int width,
                             int height)
{
    int *d_samplesPerPixel = nullptr;
    int *d_intersectionCount = nullptr;
    float *d_image = nullptr;
    unsigned char *d_finalImage = nullptr;
    unsigned char *d_finalIntersectionCountImage = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_samplesPerPixel, sizeof(int) * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_intersectionCount, sizeof(int) * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_image, sizeof(float) * 3 * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_finalImage, sizeof(unsigned char) * 3 * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_finalIntersectionCountImage, sizeof(unsigned char) * 3 * width * height));

    CHECK_CUDA(cudaMemcpy(d_samplesPerPixel, samplesPerPixel, sizeof(int) * width * height, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_intersectionCount, intersectionCount, sizeof(int) * width * height, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_image, image, sizeof(float) * 3 * width * height, cudaMemcpyHostToDevice));

    int size = width * height;
    update_final_image_kernel<256><<<dim3((size - 1) / 256 + 1), dim3(256), 0, 0>>>(
        d_image, d_samplesPerPixel, d_intersectionCount, d_finalImage, d_finalIntersectionCountImage, width, height);

    CHECK_CUDA(cudaMemcpy(finalImage, d_finalImage, sizeof(unsigned char) * 3 * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(finalIntersectionCountImage, d_finalIntersectionCountImage,
                          sizeof(unsigned char) * 3 * width * height, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_samplesPerPixel));
    CHECK_CUDA(cudaFree(d_intersectionCount));
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_finalImage));
    CHECK_CUDA(cudaFree(d_finalIntersectionCountImage));
}