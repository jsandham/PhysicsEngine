layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    vec3 cameraPos;
}Camera;