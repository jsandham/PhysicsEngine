#version 430 core
#include "camera.glsl"
#include "light.glsl"

uniform mat4 model;
in vec3 position;
in vec3 normal;
in vec2 texCoord;
out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec2 TexCoord;
out float ClipSpaceZ;
out vec4 FragPosLightSpace[5];
void main()
{
    CameraPos = Camera.cameraPos;
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
    ClipSpaceZ = gl_Position.z;
    for (int i = 0; i < 5; i++)
    {
        FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);
    }
};