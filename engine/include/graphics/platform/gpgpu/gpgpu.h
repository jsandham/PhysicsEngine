#pragma once

#include "../../../core/glm.h"
#include "../../../core/RTGeometry.h"

namespace gpgpu // rename to compute??
{
void clearPixels(float *image, int* samplesPerPixel, int* intersectionCount, int width, int height);
void updateFinalImage(const float *image, const int *samplesPerPixel, const int *intersectionCount,
                      unsigned char *finalImage, unsigned char *finalIntersectionCountImage, int width, int height);
void raytraceNormals(const PhysicsEngine::RTGeometry &geometry, float *image, int *samplesPerPixel, int *intersectionCount,
                     glm::vec3 cameraPosition, glm::mat4 projectionMatrix, int width, int height);
}