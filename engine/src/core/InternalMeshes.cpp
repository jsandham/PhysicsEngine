#include "../../include/core/InternalMeshes.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

const std::vector<float> InternalMeshes::sphereVertices = {
    0.0f,     0.5f,    -0.0f,    0.125f,   0.433f,  0.2165f,  0.25f,   0.433f,  -0.0f,    0.25f,    0.433f,  -0.0f,
    0.125f,   0.433f,  0.2165f,  0.2165f,  0.25f,   0.375f,   0.25f,   0.433f,  -0.0f,    0.2165f,  0.25f,   0.375f,
    0.433f,   0.25f,   -0.0f,    0.433f,   0.25f,   -0.0f,    0.2165f, 0.25f,   0.375f,   0.25f,    0.0f,    0.433f,
    0.433f,   0.25f,   -0.0f,    0.25f,    0.0f,    0.433f,   0.5f,    0.0f,    0.0f,     0.5f,     0.0f,    0.0f,
    0.25f,    0.0f,    0.433f,   0.2165f,  -0.25f,  0.375f,   0.5f,    0.0f,    0.0f,     0.2165f,  -0.25f,  0.375f,
    0.433f,   -0.25f,  0.0f,     0.433f,   -0.25f,  0.0f,     0.2165f, -0.25f,  0.375f,   0.125f,   -0.433f, 0.2165f,
    0.433f,   -0.25f,  0.0f,     0.125f,   -0.433f, 0.2165f,  0.25f,   -0.433f, 0.0f,     0.25f,    -0.433f, 0.0f,
    0.125f,   -0.433f, 0.2165f,  0.0f,     -0.5f,   0.0f,     0.0f,    0.5f,    -0.0f,    -0.125f,  0.433f,  0.2165f,
    0.125f,   0.433f,  0.2165f,  0.125f,   0.433f,  0.2165f,  -0.125f, 0.433f,  0.2165f,  -0.2165f, 0.25f,   0.375f,
    0.125f,   0.433f,  0.2165f,  -0.2165f, 0.25f,   0.375f,   0.2165f, 0.25f,   0.375f,   0.2165f,  0.25f,   0.375f,
    -0.2165f, 0.25f,   0.375f,   -0.25f,   0.0f,    0.433f,   0.2165f, 0.25f,   0.375f,   -0.25f,   0.0f,    0.433f,
    0.25f,    0.0f,    0.433f,   0.25f,    0.0f,    0.433f,   -0.25f,  0.0f,    0.433f,   -0.2165f, -0.25f,  0.375f,
    0.25f,    0.0f,    0.433f,   -0.2165f, -0.25f,  0.375f,   0.2165f, -0.25f,  0.375f,   0.2165f,  -0.25f,  0.375f,
    -0.2165f, -0.25f,  0.375f,   -0.125f,  -0.433f, 0.2165f,  0.2165f, -0.25f,  0.375f,   -0.125f,  -0.433f, 0.2165f,
    0.125f,   -0.433f, 0.2165f,  0.125f,   -0.433f, 0.2165f,  -0.125f, -0.433f, 0.2165f,  0.0f,     -0.5f,   0.0f,
    0.0f,     0.5f,    -0.0f,    -0.25f,   0.433f,  -0.0f,    -0.125f, 0.433f,  0.2165f,  -0.125f,  0.433f,  0.2165f,
    -0.25f,   0.433f,  -0.0f,    -0.433f,  0.25f,   -0.0f,    -0.125f, 0.433f,  0.2165f,  -0.433f,  0.25f,   -0.0f,
    -0.2165f, 0.25f,   0.375f,   -0.2165f, 0.25f,   0.375f,   -0.433f, 0.25f,   -0.0f,    -0.5f,    0.0f,    0.0f,
    -0.2165f, 0.25f,   0.375f,   -0.5f,    0.0f,    0.0f,     -0.25f,  0.0f,    0.433f,   -0.25f,   0.0f,    0.433f,
    -0.5f,    0.0f,    0.0f,     -0.433f,  -0.25f,  0.0f,     -0.25f,  0.0f,    0.433f,   -0.433f,  -0.25f,  0.0f,
    -0.2165f, -0.25f,  0.375f,   -0.2165f, -0.25f,  0.375f,   -0.433f, -0.25f,  0.0f,     -0.25f,   -0.433f, 0.0f,
    -0.2165f, -0.25f,  0.375f,   -0.25f,   -0.433f, 0.0f,     -0.125f, -0.433f, 0.2165f,  -0.125f,  -0.433f, 0.2165f,
    -0.25f,   -0.433f, 0.0f,     0.0f,     -0.5f,   0.0f,     0.0f,    0.5f,    -0.0f,    -0.125f,  0.433f,  -0.2165f,
    -0.25f,   0.433f,  -0.0f,    -0.25f,   0.433f,  -0.0f,    -0.125f, 0.433f,  -0.2165f, -0.2165f, 0.25f,   -0.375f,
    -0.25f,   0.433f,  -0.0f,    -0.2165f, 0.25f,   -0.375f,  -0.433f, 0.25f,   -0.0f,    -0.433f,  0.25f,   -0.0f,
    -0.2165f, 0.25f,   -0.375f,  -0.25f,   -0.0f,   -0.433f,  -0.433f, 0.25f,   -0.0f,    -0.25f,   -0.0f,   -0.433f,
    -0.5f,    0.0f,    0.0f,     -0.5f,    0.0f,    0.0f,     -0.25f,  -0.0f,   -0.433f,  -0.2165f, -0.25f,  -0.375f,
    -0.5f,    0.0f,    0.0f,     -0.2165f, -0.25f,  -0.375f,  -0.433f, -0.25f,  0.0f,     -0.433f,  -0.25f,  0.0f,
    -0.2165f, -0.25f,  -0.375f,  -0.125f,  -0.433f, -0.2165f, -0.433f, -0.25f,  0.0f,     -0.125f,  -0.433f, -0.2165f,
    -0.25f,   -0.433f, 0.0f,     -0.25f,   -0.433f, 0.0f,     -0.125f, -0.433f, -0.2165f, 0.0f,     -0.5f,   0.0f,
    0.0f,     0.5f,    -0.0f,    0.125f,   0.433f,  -0.2165f, -0.125f, 0.433f,  -0.2165f, -0.125f,  0.433f,  -0.2165f,
    0.125f,   0.433f,  -0.2165f, 0.2165f,  0.25f,   -0.375f,  -0.125f, 0.433f,  -0.2165f, 0.2165f,  0.25f,   -0.375f,
    -0.2165f, 0.25f,   -0.375f,  -0.2165f, 0.25f,   -0.375f,  0.2165f, 0.25f,   -0.375f,  0.25f,    -0.0f,   -0.433f,
    -0.2165f, 0.25f,   -0.375f,  0.25f,    -0.0f,   -0.433f,  -0.25f,  -0.0f,   -0.433f,  -0.25f,   -0.0f,   -0.433f,
    0.25f,    -0.0f,   -0.433f,  0.2165f,  -0.25f,  -0.375f,  -0.25f,  -0.0f,   -0.433f,  0.2165f,  -0.25f,  -0.375f,
    -0.2165f, -0.25f,  -0.375f,  -0.2165f, -0.25f,  -0.375f,  0.2165f, -0.25f,  -0.375f,  0.125f,   -0.433f, -0.2165f,
    -0.2165f, -0.25f,  -0.375f,  0.125f,   -0.433f, -0.2165f, -0.125f, -0.433f, -0.2165f, -0.125f,  -0.433f, -0.2165f,
    0.125f,   -0.433f, -0.2165f, 0.0f,     -0.5f,   0.0f,     0.0f,    0.5f,    -0.0f,    0.25f,    0.433f,  -0.0f,
    0.125f,   0.433f,  -0.2165f, 0.125f,   0.433f,  -0.2165f, 0.25f,   0.433f,  -0.0f,    0.433f,   0.25f,   -0.0f,
    0.125f,   0.433f,  -0.2165f, 0.433f,   0.25f,   -0.0f,    0.2165f, 0.25f,   -0.375f,  0.2165f,  0.25f,   -0.375f,
    0.433f,   0.25f,   -0.0f,    0.5f,     0.0f,    0.0f,     0.2165f, 0.25f,   -0.375f,  0.5f,     0.0f,    0.0f,
    0.25f,    -0.0f,   -0.433f,  0.25f,    -0.0f,   -0.433f,  0.5f,    0.0f,    0.0f,     0.433f,   -0.25f,  0.0f,
    0.25f,    -0.0f,   -0.433f,  0.433f,   -0.25f,  0.0f,     0.2165f, -0.25f,  -0.375f,  0.2165f,  -0.25f,  -0.375f,
    0.433f,   -0.25f,  0.0f,     0.25f,    -0.433f, 0.0f,     0.2165f, -0.25f,  -0.375f,  0.25f,    -0.433f, 0.0f,
    0.125f,   -0.433f, -0.2165f, 0.125f,   -0.433f, -0.2165f, 0.25f,   -0.433f, 0.0f,     0.0f,     -0.5f,   0.0f};

const std::vector<float> InternalMeshes::sphereNormals = {
    0.256f,   0.9553f,  0.1478f,  0.256f,   0.9553f,  0.1478f,  0.256f,   0.9553f,  0.1478f,  0.6546f,  0.6547f,
    0.378f,   0.6546f,  0.6547f,  0.378f,   0.6546f,  0.6547f,  0.378f,   0.6546f,  0.6547f,  0.378f,   0.6546f,
    0.6547f,  0.378f,   0.6546f,  0.6547f,  0.378f,   0.8436f,  0.2261f,  0.4871f,  0.8436f,  0.2261f,  0.4871f,
    0.8436f,  0.2261f,  0.4871f,  0.8436f,  0.2261f,  0.4871f,  0.8436f,  0.2261f,  0.4871f,  0.8436f,  0.2261f,
    0.4871f,  0.8436f,  -0.2261f, 0.4871f,  0.8436f,  -0.2261f, 0.4871f,  0.8436f,  -0.2261f, 0.4871f,  0.8436f,
    -0.2261f, 0.4871f,  0.8436f,  -0.2261f, 0.4871f,  0.8436f,  -0.2261f, 0.4871f,  0.6546f,  -0.6547f, 0.378f,
    0.6546f,  -0.6547f, 0.378f,   0.6546f,  -0.6547f, 0.378f,   0.6546f,  -0.6547f, 0.378f,   0.6546f,  -0.6547f,
    0.378f,   0.6546f,  -0.6547f, 0.378f,   0.256f,   -0.9553f, 0.1478f,  0.256f,   -0.9553f, 0.1478f,  0.256f,
    -0.9553f, 0.1478f,  0.0f,     0.9553f,  0.2956f,  0.0f,     0.9553f,  0.2956f,  0.0f,     0.9553f,  0.2956f,
    0.0f,     0.6547f,  0.7559f,  0.0f,     0.6547f,  0.7559f,  0.0f,     0.6547f,  0.7559f,  0.0f,     0.6547f,
    0.7559f,  0.0f,     0.6547f,  0.7559f,  0.0f,     0.6547f,  0.7559f,  0.0f,     0.226f,   0.9741f,  0.0f,
    0.226f,   0.9741f,  0.0f,     0.226f,   0.9741f,  0.0f,     0.226f,   0.9741f,  0.0f,     0.226f,   0.9741f,
    0.0f,     0.226f,   0.9741f,  0.0f,     -0.226f,  0.9741f,  0.0f,     -0.226f,  0.9741f,  0.0f,     -0.226f,
    0.9741f,  0.0f,     -0.226f,  0.9741f,  0.0f,     -0.226f,  0.9741f,  0.0f,     -0.226f,  0.9741f,  0.0f,
    -0.6547f, 0.7559f,  0.0f,     -0.6547f, 0.7559f,  0.0f,     -0.6547f, 0.7559f,  0.0f,     -0.6547f, 0.7559f,
    0.0f,     -0.6547f, 0.7559f,  0.0f,     -0.6547f, 0.7559f,  0.0f,     -0.9553f, 0.2956f,  0.0f,     -0.9553f,
    0.2956f,  0.0f,     -0.9553f, 0.2956f,  -0.256f,  0.9553f,  0.1478f,  -0.256f,  0.9553f,  0.1478f,  -0.256f,
    0.9553f,  0.1478f,  -0.6546f, 0.6547f,  0.378f,   -0.6546f, 0.6547f,  0.378f,   -0.6546f, 0.6547f,  0.378f,
    -0.6546f, 0.6547f,  0.378f,   -0.6546f, 0.6547f,  0.378f,   -0.6546f, 0.6547f,  0.378f,   -0.8436f, 0.2261f,
    0.4871f,  -0.8436f, 0.2261f,  0.4871f,  -0.8436f, 0.2261f,  0.4871f,  -0.8436f, 0.2261f,  0.4871f,  -0.8436f,
    0.2261f,  0.4871f,  -0.8436f, 0.2261f,  0.4871f,  -0.8436f, -0.2261f, 0.4871f,  -0.8436f, -0.2261f, 0.4871f,
    -0.8436f, -0.2261f, 0.4871f,  -0.8436f, -0.2261f, 0.4871f,  -0.8436f, -0.2261f, 0.4871f,  -0.8436f, -0.2261f,
    0.4871f,  -0.6546f, -0.6547f, 0.378f,   -0.6546f, -0.6547f, 0.378f,   -0.6546f, -0.6547f, 0.378f,   -0.6546f,
    -0.6547f, 0.378f,   -0.6546f, -0.6547f, 0.378f,   -0.6546f, -0.6547f, 0.378f,   -0.256f,  -0.9553f, 0.1478f,
    -0.256f,  -0.9553f, 0.1478f,  -0.256f,  -0.9553f, 0.1478f,  -0.256f,  0.9553f,  -0.1478f, -0.256f,  0.9553f,
    -0.1478f, -0.256f,  0.9553f,  -0.1478f, -0.6546f, 0.6547f,  -0.378f,  -0.6546f, 0.6547f,  -0.378f,  -0.6546f,
    0.6547f,  -0.378f,  -0.6546f, 0.6547f,  -0.378f,  -0.6546f, 0.6547f,  -0.378f,  -0.6546f, 0.6547f,  -0.378f,
    -0.8436f, 0.2261f,  -0.4871f, -0.8436f, 0.2261f,  -0.4871f, -0.8436f, 0.2261f,  -0.4871f, -0.8436f, 0.2261f,
    -0.4871f, -0.8436f, 0.2261f,  -0.4871f, -0.8436f, 0.2261f,  -0.4871f, -0.8436f, -0.2261f, -0.4871f, -0.8436f,
    -0.2261f, -0.4871f, -0.8436f, -0.2261f, -0.4871f, -0.8436f, -0.2261f, -0.4871f, -0.8436f, -0.2261f, -0.4871f,
    -0.8436f, -0.2261f, -0.4871f, -0.6546f, -0.6547f, -0.378f,  -0.6546f, -0.6547f, -0.378f,  -0.6546f, -0.6547f,
    -0.378f,  -0.6546f, -0.6547f, -0.378f,  -0.6546f, -0.6547f, -0.378f,  -0.6546f, -0.6547f, -0.378f,  -0.256f,
    -0.9553f, -0.1478f, -0.256f,  -0.9553f, -0.1478f, -0.256f,  -0.9553f, -0.1478f, 0.0f,     0.9553f,  -0.2956f,
    0.0f,     0.9553f,  -0.2956f, 0.0f,     0.9553f,  -0.2956f, 0.0f,     0.6547f,  -0.7559f, 0.0f,     0.6547f,
    -0.7559f, 0.0f,     0.6547f,  -0.7559f, 0.0f,     0.6547f,  -0.7559f, 0.0f,     0.6547f,  -0.7559f, 0.0f,
    0.6547f,  -0.7559f, 0.0f,     0.226f,   -0.9741f, 0.0f,     0.226f,   -0.9741f, 0.0f,     0.226f,   -0.9741f,
    0.0f,     0.226f,   -0.9741f, 0.0f,     0.226f,   -0.9741f, 0.0f,     0.226f,   -0.9741f, 0.0f,     -0.226f,
    -0.9741f, 0.0f,     -0.226f,  -0.9741f, 0.0f,     -0.226f,  -0.9741f, 0.0f,     -0.226f,  -0.9741f, 0.0f,
    -0.226f,  -0.9741f, 0.0f,     -0.226f,  -0.9741f, 0.0f,     -0.6547f, -0.7559f, 0.0f,     -0.6547f, -0.7559f,
    0.0f,     -0.6547f, -0.7559f, 0.0f,     -0.6547f, -0.7559f, 0.0f,     -0.6547f, -0.7559f, 0.0f,     -0.6547f,
    -0.7559f, 0.0f,     -0.9553f, -0.2956f, 0.0f,     -0.9553f, -0.2956f, 0.0f,     -0.9553f, -0.2956f, 0.256f,
    0.9553f,  -0.1478f, 0.256f,   0.9553f,  -0.1478f, 0.256f,   0.9553f,  -0.1478f, 0.6546f,  0.6547f,  -0.378f,
    0.6546f,  0.6547f,  -0.378f,  0.6546f,  0.6547f,  -0.378f,  0.6546f,  0.6547f,  -0.378f,  0.6546f,  0.6547f,
    -0.378f,  0.6546f,  0.6547f,  -0.378f,  0.8436f,  0.2261f,  -0.4871f, 0.8436f,  0.2261f,  -0.4871f, 0.8436f,
    0.2261f,  -0.4871f, 0.8436f,  0.2261f,  -0.4871f, 0.8436f,  0.2261f,  -0.4871f, 0.8436f,  0.2261f,  -0.4871f,
    0.8436f,  -0.2261f, -0.4871f, 0.8436f,  -0.2261f, -0.4871f, 0.8436f,  -0.2261f, -0.4871f, 0.8436f,  -0.2261f,
    -0.4871f, 0.8436f,  -0.2261f, -0.4871f, 0.8436f,  -0.2261f, -0.4871f, 0.6546f,  -0.6547f, -0.378f,  0.6546f,
    -0.6547f, -0.378f,  0.6546f,  -0.6547f, -0.378f,  0.6546f,  -0.6547f, -0.378f,  0.6546f,  -0.6547f, -0.378f,
    0.6546f,  -0.6547f, -0.378f,  0.256f,   -0.9553f, -0.1478f, 0.256f,   -0.9553f, -0.1478f, 0.256f,   -0.9553f,
    -0.1478f};

const std::vector<float> InternalMeshes::sphereTexCoords = {
    0.0f,      0.0f,      0.166667f, 0.166667f, 0.0f,      0.166667f, 0.0f,      0.166667f, 0.166667f, 0.166667f,
    0.166667f, 0.333333f, 0.0f,      0.166667f, 0.166667f, 0.333333f, 0.0f,      0.333333f, 0.0f,      0.333333f,
    0.166667f, 0.333333f, 0.166667f, 0.5f,      0.0f,      0.333333f, 0.166667f, 0.5f,      0.0f,      0.5f,
    0.0f,      0.5f,      0.166667f, 0.5f,      0.166667f, 0.666667f, 0.0f,      0.5f,      0.166667f, 0.666667f,
    0.0f,      0.666667f, 0.0f,      0.666667f, 0.166667f, 0.666667f, 0.166667f, 0.833333f, 0.0f,      0.666667f,
    0.166667f, 0.833333f, 0.0f,      0.833333f, 0.0f,      0.833333f, 0.166667f, 0.833333f, 0.166667f, 1.0f,
    0.166667f, 0.0f,      0.333333f, 0.166667f, 0.166667f, 0.166667f, 0.166667f, 0.166667f, 0.333333f, 0.166667f,
    0.333333f, 0.333333f, 0.166667f, 0.166667f, 0.333333f, 0.333333f, 0.166667f, 0.333333f, 0.166667f, 0.333333f,
    0.333333f, 0.333333f, 0.333333f, 0.5f,      0.166667f, 0.333333f, 0.333333f, 0.5f,      0.166667f, 0.5f,
    0.166667f, 0.5f,      0.333333f, 0.5f,      0.333333f, 0.666667f, 0.166667f, 0.5f,      0.333333f, 0.666667f,
    0.166667f, 0.666667f, 0.166667f, 0.666667f, 0.333333f, 0.666667f, 0.333333f, 0.833333f, 0.166667f, 0.666667f,
    0.333333f, 0.833333f, 0.166667f, 0.833333f, 0.166667f, 0.833333f, 0.333333f, 0.833333f, 0.333333f, 1.0f,
    0.333333f, 0.0f,      0.5f,      0.166667f, 0.333333f, 0.166667f, 0.333333f, 0.166667f, 0.5f,      0.166667f,
    0.5f,      0.333333f, 0.333333f, 0.166667f, 0.5f,      0.333333f, 0.333333f, 0.333333f, 0.333333f, 0.333333f,
    0.5f,      0.333333f, 0.5f,      0.5f,      0.333333f, 0.333333f, 0.5f,      0.5f,      0.333333f, 0.5f,
    0.333333f, 0.5f,      0.5f,      0.5f,      0.5f,      0.666667f, 0.333333f, 0.5f,      0.5f,      0.666667f,
    0.333333f, 0.666667f, 0.333333f, 0.666667f, 0.5f,      0.666667f, 0.5f,      0.833333f, 0.333333f, 0.666667f,
    0.5f,      0.833333f, 0.333333f, 0.833333f, 0.333333f, 0.833333f, 0.5f,      0.833333f, 0.5f,      1.0f,
    0.5f,      0.0f,      0.666667f, 0.166667f, 0.5f,      0.166667f, 0.5f,      0.166667f, 0.666667f, 0.166667f,
    0.666667f, 0.333333f, 0.5f,      0.166667f, 0.666667f, 0.333333f, 0.5f,      0.333333f, 0.5f,      0.333333f,
    0.666667f, 0.333333f, 0.666667f, 0.5f,      0.5f,      0.333333f, 0.666667f, 0.5f,      0.5f,      0.5f,
    0.5f,      0.5f,      0.666667f, 0.5f,      0.666667f, 0.666667f, 0.5f,      0.5f,      0.666667f, 0.666667f,
    0.5f,      0.666667f, 0.5f,      0.666667f, 0.666667f, 0.666667f, 0.666667f, 0.833333f, 0.5f,      0.666667f,
    0.666667f, 0.833333f, 0.5f,      0.833333f, 0.5f,      0.833333f, 0.666667f, 0.833333f, 0.666667f, 1.0f,
    0.666667f, 0.0f,      0.833333f, 0.166667f, 0.666667f, 0.166667f, 0.666667f, 0.166667f, 0.833333f, 0.166667f,
    0.833333f, 0.333333f, 0.666667f, 0.166667f, 0.833333f, 0.333333f, 0.666667f, 0.333333f, 0.666667f, 0.333333f,
    0.833333f, 0.333333f, 0.833333f, 0.5f,      0.666667f, 0.333333f, 0.833333f, 0.5f,      0.666667f, 0.5f,
    0.666667f, 0.5f,      0.833333f, 0.5f,      0.833333f, 0.666667f, 0.666667f, 0.5f,      0.833333f, 0.666667f,
    0.666667f, 0.666667f, 0.666667f, 0.666667f, 0.833333f, 0.666667f, 0.833333f, 0.833333f, 0.666667f, 0.666667f,
    0.833333f, 0.833333f, 0.666667f, 0.833333f, 0.666667f, 0.833333f, 0.833333f, 0.833333f, 0.833333f, 1.0f,
    0.833333f, 0.0f,      1.0f,      0.166667f, 0.833333f, 0.166667f, 0.833333f, 0.166667f, 1.0f,      0.166667f,
    1.0f,      0.333333f, 0.833333f, 0.166667f, 1.0f,      0.333333f, 0.833333f, 0.333333f, 0.833333f, 0.333333f,
    1.0f,      0.333333f, 1.0f,      0.5f,      0.833333f, 0.333333f, 1.0f,      0.5f,      0.833333f, 0.5f,
    0.833333f, 0.5f,      1.0f,      0.5f,      1.0f,      0.666667f, 0.833333f, 0.5f,      1.0f,      0.666667f,
    0.833333f, 0.666667f, 0.833333f, 0.666667f, 1.0f,      0.666667f, 1.0f,      0.833333f, 0.833333f, 0.666667f,
    1.0f,      0.833333f, 0.833333f, 0.833333f, 0.833333f, 0.833333f, 1.0f,      0.833333f, 1.0f,      1.0f};

const std::vector<int> InternalMeshes::sphereSubMeshStartIndicies = {0, 540};

const std::vector<float> InternalMeshes::cubeVertices = {-0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f,
                                                         0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,

                                                         -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
                                                         0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,

                                                         -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,
                                                         -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,

                                                         0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f,
                                                         0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,

                                                         -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,
                                                         0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f,

                                                         -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,
                                                         0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f};

const std::vector<float> InternalMeshes::cubeNormals = {0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f,
                                                        0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f,

                                                        0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
                                                        0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,

                                                        -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,
                                                        -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,

                                                        1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
                                                        1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

                                                        0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,
                                                        0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,  0.0f,  -1.0f, 0.0f,

                                                        0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
                                                        0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f};

const std::vector<float> InternalMeshes::cubeTexCoords = {
    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,

    0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,

    1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

    1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

    0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,

    0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

const std::vector<int> InternalMeshes::cubeSubMeshStartIndicies = {0, 108};

const std::vector<float> InternalMeshes::planeVertices = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
                                                          -1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f};

const std::vector<float> InternalMeshes::planeNormals = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                                         0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};

const std::vector<float> InternalMeshes::planeTexCoords = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
                                                           0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

const std::vector<int> InternalMeshes::planeSubMeshStartIndicies = {0, 18};

const Guid InternalMeshes::sphereMeshId("0a79687e-6398-4a50-9187-4387b9098bef");
const Guid InternalMeshes::cubeMeshId("4d267f0a-bacf-403e-9381-b7b313f609f6");
const Guid InternalMeshes::planeMeshId("4899cc8c-ec05-4b99-9dfe-908dfb17359d");

Guid InternalMeshes::loadInternalMesh(World *world, const Guid meshId, const std::vector<float> &vertices,
                                      const std::vector<float> &normals, const std::vector<float> &texCoords,
                                      const std::vector<int> &startIndices)
{
    // Create temp mesh to compute serialized data vector
    Mesh temp;
    temp.load(vertices, normals, texCoords, startIndices);

    std::vector<char> data = temp.serialize(meshId);

    Mesh *mesh = world->createAsset<Mesh>(data);
    if (mesh != NULL)
    {
        return mesh->getId();
    }
    else
    {
        Log::error("Could not create internal mesh\n");
        return Guid::INVALID;
    }
}

Guid InternalMeshes::loadSphereMesh(World *world)
{
    return loadInternalMesh(world, InternalMeshes::sphereMeshId, InternalMeshes::sphereVertices,
                            InternalMeshes::sphereNormals, InternalMeshes::sphereTexCoords,
                            InternalMeshes::sphereSubMeshStartIndicies);
}

Guid InternalMeshes::loadCubeMesh(World *world)
{
    return loadInternalMesh(world, InternalMeshes::cubeMeshId, InternalMeshes::cubeVertices,
                            InternalMeshes::cubeNormals, InternalMeshes::cubeTexCoords,
                            InternalMeshes::cubeSubMeshStartIndicies);
}

Guid InternalMeshes::loadPlaneMesh(World *world)
{
    return loadInternalMesh(world, InternalMeshes::planeMeshId, InternalMeshes::planeVertices,
                            InternalMeshes::planeNormals, InternalMeshes::planeTexCoords,
                            InternalMeshes::planeSubMeshStartIndicies);
}