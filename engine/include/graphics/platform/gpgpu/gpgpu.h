#pragma once

namespace gpgpu // rename to compute??
{
void clearPixels(float *image, int* samplesPerPixel, int* intersectionCount, int width, int height);
void updateFinalImage(const float *image, const int *samplesPerPixel, const int *intersectionCount,
                      unsigned char *finalImage, unsigned char *finalIntersectionCountImage, int width, int height);
}