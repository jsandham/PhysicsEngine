
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

    __host__ __device__ __forceinline__ vec2() : x(0.0f), y(0.0f){};
    __host__ __device__ __forceinline__ vec2(float x, float y) : x(x), y(y){};
    __host__ __device__ __forceinline__ vec2 operator+=(const vec2 &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        return *this;
    }

    __host__ __device__ __forceinline__ vec2 operator-=(const vec2 &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        return *this;
    }

    __host__ __device__ __forceinline__ vec2 operator*=(float scalar)
    {
        this->x *= scalar;
        this->y *= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ vec2 operator*=(const vec2 &vec)
    {
        this->x *= vec.x;
        this->y *= vec.y;
        return *this;
    }

    __host__ __device__ __forceinline__ vec2 operator/=(const vec2 &vec)
    {
        this->x /= vec.x;
        this->y /= vec.y;
        return *this;
    }
};

__host__ __device__ __forceinline__ vec2 operator*(const vec2 &v, float scalar)
{
    return vec2(v.x * scalar, v.y * scalar);
}

__host__ __device__ __forceinline__ vec2 operator+(const vec2 &v1, const vec2 &v2)
{
    return vec2(v1.x + v2.x, v1.y + v2.y);
}

__host__ __device__ __forceinline__ vec2 operator-(const vec2 &v1, const vec2 &v2)
{
    return vec2(v1.x - v2.x, v1.y - v2.y);
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

    __host__ __device__ __forceinline__ vec3() : x(0.0f), y(0.0f), z(0.0f){};
    __host__ __device__ __forceinline__ vec3(float x, float y, float z) : x(x), y(y), z(z){};

    __host__ __device__ __forceinline__ vec3 operator+=(const vec3 &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
        return *this;
    }

    __host__ __device__ __forceinline__ vec3 operator-=(const vec3 &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        this->z -= vec.z;
        return *this;
    }

    __host__ __device__ __forceinline__ vec3 operator*=(float scalar)
    {
        this->x *= scalar;
        this->y *= scalar;
        this->z *= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ vec3 operator*=(const vec3 &vec)
    {
        this->x *= vec.x;
        this->y *= vec.y;
        this->z *= vec.z;
        return *this;
    }

    __host__ __device__ __forceinline__ vec3 operator/=(const vec3 &vec)
    {
        this->x /= vec.x;
        this->y /= vec.y;
        this->z /= vec.z;
        return *this;
    }
};

__host__ __device__ __forceinline__ vec3 operator*(const vec3 &v, float scalar)
{
    return vec3(v.x * scalar, v.y * scalar, v.z * scalar);
}

__host__ __device__ __forceinline__ vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__host__ __device__ __forceinline__ vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

struct vec4
{
    union {
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            float r, g, b, a;
        };
    };

    __host__ __device__ __forceinline__ vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f){};
    __host__ __device__ __forceinline__ vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w){};

    __host__ __device__ __forceinline__ vec4 operator+=(const vec4 &vec)
    {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
        this->w += vec.w;
        return *this;
    }

    __host__ __device__ __forceinline__ vec4 operator-=(const vec4 &vec)
    {
        this->x -= vec.x;
        this->y -= vec.y;
        this->z -= vec.z;
        this->w -= vec.w;
        return *this;
    }

    __host__ __device__ __forceinline__ vec4 operator*=(float scalar)
    {
        this->x *= scalar;
        this->y *= scalar;
        this->z *= scalar;
        this->w *= scalar;
        return *this;
    }

    __host__ __device__ __forceinline__ vec4 operator*=(const vec4 &vec)
    {
        this->x *= vec.x;
        this->y *= vec.y;
        this->z *= vec.z;
        this->w *= vec.w;
        return *this;
    }

    __host__ __device__ __forceinline__ vec4 operator/=(const vec4 &vec)
    {
        this->x /= vec.x;
        this->y /= vec.y;
        this->z /= vec.z;
        this->w /= vec.w;
        return *this;
    }
};

__host__ __device__ __forceinline__ vec4 operator*(const vec4 &v, float scalar)
{
    return vec4(v.x * scalar, v.y * scalar, v.z * scalar, v.w * scalar);
}

__host__ __device__ __forceinline__ vec4 operator+(const vec4 &v1, const vec4 &v2)
{
    return vec4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
}

__host__ __device__ __forceinline__ vec4 operator-(const vec4 &v1, const vec4 &v2)
{
    return vec4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
}






// mat4 code taken from https://gist.github.com/mattatz
// Not sure how to credit or indicate license for this.
struct mat4
{
    float m[4][4];

    __host__ __device__ __forceinline__ mat4()
    {
        m[0][0] = 1.0;
        m[1][0] = 0.0;
        m[2][0] = 0.0;
        m[3][0] = 0.0;
        m[0][1] = 0.0;
        m[1][1] = 1.0;
        m[2][1] = 0.0;
        m[3][1] = 0.0;
        m[0][2] = 0.0;
        m[1][2] = 0.0;
        m[2][2] = 1.0;
        m[3][2] = 0.0;
        m[0][3] = 0.0;
        m[1][3] = 0.0;
        m[2][3] = 0.0;
        m[3][3] = 1.0;
    }

    __host__ __device__ __forceinline__ mat4(const float m11, const float m12, const float m13, const float m14,
                                             const float m21, const float m22, const float m23, const float m24,
                                             const float m31, const float m32, const float m33, const float m34,
                                             const float m41, const float m42, const float m43, const float m44)
    {
        m[0][0] = m11;
        m[1][0] = m12;
        m[2][0] = m13;
        m[3][0] = m14;
        m[0][1] = m21;
        m[1][1] = m22;
        m[2][1] = m23;
        m[3][1] = m24;
        m[0][2] = m31;
        m[1][2] = m32;
        m[2][2] = m33;
        m[3][2] = m34;
        m[0][3] = m41;
        m[1][3] = m42;
        m[2][3] = m43;
        m[3][3] = m44;
    }

    __host__ __device__ __forceinline__ float *operator[](const size_t idx)
    {
        return m[idx];
    }

    __host__ __device__ __forceinline__ float4 operator*(const float4 &v) const
    {
        float4 ret;
        ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w;
        ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w;
        ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w;
        ret.w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w;
        return ret;
    }

    // Added this myself
    __host__ __device__ __forceinline__ vec4 operator*(const vec4 &v) const
    {
        vec4 ret;
        ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w;
        ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w;
        ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w;
        ret.w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const float f) const
    {
        mat4 ret;
        ret[0][0] = m[0][0] * f;
        ret[1][0] = m[1][0] * f;
        ret[2][0] = m[2][0] * f;
        ret[3][0] = m[3][0] * f;
        ret[0][1] = m[0][1] * f;
        ret[1][1] = m[1][1] * f;
        ret[2][1] = m[2][1] * f;
        ret[3][1] = m[3][1] * f;
        ret[0][2] = m[0][2] * f;
        ret[1][2] = m[1][2] * f;
        ret[2][2] = m[2][2] * f;
        ret[3][2] = m[3][2] * f;
        ret[0][3] = m[0][3] * f;
        ret[1][3] = m[1][3] * f;
        ret[2][3] = m[2][3] * f;
        ret[3][3] = m[3][3] * f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator/(const float f) const
    {
        mat4 ret;
        ret[0][0] = m[0][0] / f;
        ret[1][0] = m[1][0] / f;
        ret[2][0] = m[2][0] / f;
        ret[3][0] = m[3][0] / f;
        ret[0][1] = m[0][1] / f;
        ret[1][1] = m[1][1] / f;
        ret[2][1] = m[2][1] / f;
        ret[3][1] = m[3][1] / f;
        ret[0][2] = m[0][2] / f;
        ret[1][2] = m[1][2] / f;
        ret[2][2] = m[2][2] / f;
        ret[3][2] = m[3][2] / f;
        ret[0][3] = m[0][3] / f;
        ret[1][3] = m[1][3] / f;
        ret[2][3] = m[2][3] / f;
        ret[3][3] = m[3][3] / f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator+(const mat4 &other) const
    {
        mat4 ret;
        ret[0][0] = m[0][0] + other.m[0][0];
        ret[1][0] = m[1][0] + other.m[1][0];
        ret[2][0] = m[2][0] + other.m[2][0];
        ret[3][0] = m[3][0] + other.m[3][0];
        ret[0][1] = m[0][1] + other.m[0][1];
        ret[1][1] = m[1][1] + other.m[1][1];
        ret[2][1] = m[2][1] + other.m[2][1];
        ret[3][1] = m[3][1] + other.m[3][1];
        ret[0][2] = m[0][2] + other.m[0][2];
        ret[1][2] = m[1][2] + other.m[1][2];
        ret[2][2] = m[2][2] + other.m[2][2];
        ret[3][2] = m[3][2] + other.m[3][2];
        ret[0][3] = m[0][3] + other.m[0][3];
        ret[1][3] = m[1][3] + other.m[1][3];
        ret[2][3] = m[2][3] + other.m[2][3];
        ret[3][3] = m[3][3] + other.m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator-(const mat4 &other) const
    {
        mat4 ret;
        ret[0][0] = m[0][0] - other.m[0][0];
        ret[1][0] = m[1][0] - other.m[1][0];
        ret[2][0] = m[2][0] - other.m[2][0];
        ret[3][0] = m[3][0] - other.m[3][0];
        ret[0][1] = m[0][1] - other.m[0][1];
        ret[1][1] = m[1][1] - other.m[1][1];
        ret[2][1] = m[2][1] - other.m[2][1];
        ret[3][1] = m[3][1] - other.m[3][1];
        ret[0][2] = m[0][2] - other.m[0][2];
        ret[1][2] = m[1][2] - other.m[1][2];
        ret[2][2] = m[2][2] - other.m[2][2];
        ret[3][2] = m[3][2] - other.m[3][2];
        ret[0][3] = m[0][3] - other.m[0][3];
        ret[1][3] = m[1][3] - other.m[1][3];
        ret[2][3] = m[2][3] - other.m[2][3];
        ret[3][3] = m[3][3] - other.m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const mat4 &other) const
    {
        auto a11 = m[0][0], a12 = m[1][0], a13 = m[2][0], a14 = m[3][0];
        auto a21 = m[0][1], a22 = m[1][1], a23 = m[2][1], a24 = m[3][1];
        auto a31 = m[0][2], a32 = m[1][2], a33 = m[2][2], a34 = m[3][2];
        auto a41 = m[0][3], a42 = m[1][3], a43 = m[2][3], a44 = m[3][3];

        auto b11 = other.m[0][0], b12 = other.m[1][0], b13 = other.m[2][0], b14 = other.m[3][0];
        auto b21 = other.m[0][1], b22 = other.m[1][1], b23 = other.m[2][1], b24 = other.m[3][1];
        auto b31 = other.m[0][2], b32 = other.m[1][2], b33 = other.m[2][2], b34 = other.m[3][2];
        auto b41 = other.m[0][3], b42 = other.m[1][3], b43 = other.m[2][3], b44 = other.m[3][3];

        mat4 ret;
        ret[0][0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
        ret[0][1] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
        ret[0][2] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
        ret[0][3] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

        ret[1][0] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
        ret[1][1] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
        ret[1][2] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
        ret[1][3] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

        ret[2][0] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
        ret[2][1] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
        ret[2][2] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
        ret[2][3] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

        ret[3][0] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
        ret[3][1] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
        ret[3][2] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
        ret[3][3] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 transpose() const
    {
        mat4 ret;
        ret[0][0] = m[0][0];
        ret[0][1] = m[1][0];
        ret[0][2] = m[2][0];
        ret[0][3] = m[3][0];
        ret[1][0] = m[0][1];
        ret[1][1] = m[1][1];
        ret[1][2] = m[2][1];
        ret[1][3] = m[3][1];
        ret[2][0] = m[0][2];
        ret[2][1] = m[1][2];
        ret[2][2] = m[2][2];
        ret[2][3] = m[3][2];
        ret[3][0] = m[0][3];
        ret[3][1] = m[1][3];
        ret[3][2] = m[2][3];
        ret[3][3] = m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ float det() const
    {
        auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
        auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
        auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
        auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

        return (n41 * (+n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 -
                       n12 * n23 * n34) +
                n42 * (+n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 - n13 * n21 * n34 + n13 * n24 * n31 -
                       n14 * n23 * n31) +
                n43 * (+n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 + n12 * n21 * n34 + n14 * n22 * n31 -
                       n12 * n24 * n31) +
                n44 * (-n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 + n13 * n21 * n32 - n12 * n21 * n33 +
                       n12 * n23 * n31));
    }

    __host__ __device__ __forceinline__ mat4 inverse() const
    {
        auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
        auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
        auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
        auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

        auto t11 =
            n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        auto t12 =
            n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        auto t13 =
            n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        auto t14 =
            n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        auto det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
        auto idet = 1.0f / det;

        mat4 ret;

        ret[0][0] = t11 * idet;
        ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 -
                     n21 * n33 * n44) *
                    idet;
        ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 +
                     n21 * n32 * n44) *
                    idet;
        ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 -
                     n21 * n32 * n43) *
                    idet;

        ret[1][0] = t12 * idet;
        ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 +
                     n11 * n33 * n44) *
                    idet;
        ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 -
                     n11 * n32 * n44) *
                    idet;
        ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 +
                     n11 * n32 * n43) *
                    idet;

        ret[2][0] = t13 * idet;
        ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 -
                     n11 * n23 * n44) *
                    idet;
        ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 +
                     n11 * n22 * n44) *
                    idet;
        ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 -
                     n11 * n22 * n43) *
                    idet;

        ret[3][0] = t14 * idet;
        ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 +
                     n11 * n23 * n34) *
                    idet;
        ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 -
                     n11 * n22 * n34) *
                    idet;
        ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 +
                     n11 * n22 * n33) *
                    idet;

        return ret;
    }

    __host__ __device__ __forceinline__ void zero()
    {
        m[0][0] = 0.0;
        m[1][0] = 0.0;
        m[2][0] = 0.0;
        m[3][0] = 0.0;
        m[0][1] = 0.0;
        m[1][1] = 0.0;
        m[2][1] = 0.0;
        m[3][1] = 0.0;
        m[0][2] = 0.0;
        m[1][2] = 0.0;
        m[2][2] = 0.0;
        m[3][2] = 0.0;
        m[0][3] = 0.0;
        m[1][3] = 0.0;
        m[2][3] = 0.0;
        m[3][3] = 0.0;
    }

    __host__ __device__ __forceinline__ void identity()
    {
        m[0][0] = 1.0;
        m[1][0] = 0.0;
        m[2][0] = 0.0;
        m[3][0] = 0.0;
        m[0][1] = 0.0;
        m[1][1] = 1.0;
        m[2][1] = 0.0;
        m[3][1] = 0.0;
        m[0][2] = 0.0;
        m[1][2] = 0.0;
        m[2][2] = 1.0;
        m[3][2] = 0.0;
        m[0][3] = 0.0;
        m[1][3] = 0.0;
        m[2][3] = 0.0;
        m[3][3] = 1.0;
    }

    __host__ __device__ __forceinline__ mat4 &operator*=(const float f)
    {
        return *this = *this * f;
    }
    __host__ __device__ __forceinline__ mat4 &operator/=(const float f)
    {
        return *this = *this / f;
    }
    __host__ __device__ __forceinline__ mat4 &operator+=(const mat4 &m)
    {
        return *this = *this + m;
    }
    __host__ __device__ __forceinline__ mat4 &operator-=(const mat4 &m)
    {
        return *this = *this - m;
    }
    __host__ __device__ __forceinline__ mat4 &operator*=(const mat4 &m)
    {
        return *this = *this * m;
    }
};



struct Triangle
{

};








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

__device__ vec2 generatePixelSampleNDC(int u, int v, float du, float dv)
{
    // NDC coordinates for 2x2x2 cube [-1, 1]x[-1, 1]x[-1, 1]
    //         +y |   * +z
    //            |  *
    //            | *
    // -x ________|*________ +x
    // In frustum plane, bottom, left, far corner correspnds to NDC point [-1, -1, 1]
    vec2 bottomLeft_NDC = vec2(-1.0f, -1.0f);

    // Bottom, left corner pixel centre in NDC
    vec2 pixelCentreBottomLeft_NDC = bottomLeft_NDC + vec2(0.5f * du, 0.5f * dv);

    // Plane pixel centre in NDC
    vec2 pixelCentre_NDC = pixelCentreBottomLeft_NDC + vec2(u * du, v * dv);

    // Randomly sample from pixel
    return pixelCentre_NDC + vec2(generate_rand(-0.5f, 0.5f) * du, generate_rand(-0.5f, 0.5f) * dv);
}

__device__ Ray getCameraRay(vec2 pixelSampleNDC, vec3 cameraPos, mat4 inverseViewProjMatrix)
{
    // Transform NDC coordinate back to world space
    vec4 temp = inverseViewProjMatrix * vec4(pixelSampleNDC.x, pixelSampleNDC.y, 1.0f, 1.0f);
    vec3 pixelCentre_WorldSpace = vec3(temp.x / temp.w, temp.y / temp.w, temp.z / temp.w);

    Ray ray;
    ray.mOrigin = cameraPos;
    ray.mDirection = pixelCentre_WorldSpace - cameraPos;

    return ray;
}



template <unsigned int BLOCKSIZE>
__global__ void raytrace_normals_kernel(float *image, int *samplesPerPixel, int *intersectionCount, vec3 camerPos,
                                        mat4 projMatrix, int width, int height)
{
    // In NDC we use a 2x2x2 box ranging from [-1,1]x[-1,1]x[-1,1]
    float du = 2.0f / 256;
    float dv = 2.0f / 256;

    int col = 16 * blockIdx.x + threadIdx.x;
    int row = 16 * blockIdx.y + threadIdx.y;

    vec2 pixelSampleNDC = generatePixelSampleNDC(col, row, du, dv);

    int irow = (int)(height * (0.5f * (pixelSampleNDC.y + 1.0f)));
    int icol = (int)(width * (0.5f * (pixelSampleNDC.x + 1.0f)));

    irow = min(height - 1, max(0, irow));
    icol = min(width - 1, max(0, icol));

    int offset = width * irow + icol;

    samplesPerPixel[offset]++;

    // Read color from image
    float red = image[3 * offset + 0];
    float green = image[3 * offset + 1];
    float blue = image[3 * offset + 2];
    vec3 color = vec3(red, green, blue);

    int count = 0;

    if (blockIdx.y % 2 == 0 && blockIdx.x % 2 == 0)
    {
        color += vec3(1.0f, 0.0f, 0.0f);    
    }
    else
    {
        color += vec3(0.0f, 1.0f, 0.0f);
    }

    //color +=
    //    computeNormalsIterative(tlas, blas, models, getCameraRay(pixelSampleNDC), intersectionCount);

    // Store computed color to image
    image[3 * offset + 0] = color.r;
    image[3 * offset + 1] = color.g;
    image[3 * offset + 2] = color.b;

    intersectionCount[offset] += count;
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

void gpgpu::raytraceNormals(float *image, int *samplesPerPixel, int *intersectionCount, glm::vec3 cameraPosition,
                            glm::mat4 projectionMatrix, int width, int height)
{
    int *d_samplesPerPixel = nullptr;
    int *d_intersectionCount = nullptr;
    float *d_image = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_samplesPerPixel, sizeof(int) * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_intersectionCount, sizeof(int) * width * height));
    CHECK_CUDA(cudaMalloc((void **)&d_image, sizeof(float) * 3 * width * height));

    CHECK_CUDA(cudaMemcpy(d_samplesPerPixel, samplesPerPixel, sizeof(int) * width * height, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_intersectionCount, intersectionCount, sizeof(int) * width * height, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_image, image, sizeof(float) * 3 * width * height, cudaMemcpyHostToDevice));


    vec3 cameraPos;
    cameraPos.x = cameraPosition.x;
    cameraPos.y = cameraPosition.y;
    cameraPos.z = cameraPosition.z;

    mat4 projMatrix;

    dim3 grid(width / 16, height / 16, 1);
    dim3 blocks(16, 16, 1);

    raytrace_normals_kernel<256><<<grid, blocks>>>(
        d_image, d_samplesPerPixel, d_intersectionCount, cameraPos, projMatrix, width, height);




    CHECK_CUDA(cudaMemcpy(samplesPerPixel, d_samplesPerPixel, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(intersectionCount, d_intersectionCount, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(image, d_image, sizeof(float) * 3 * width * height, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_samplesPerPixel));
    CHECK_CUDA(cudaFree(d_intersectionCount));
    CHECK_CUDA(cudaFree(d_image));
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