#vertex
#version 430 core
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    mat4 viewProjection;
    vec3 cameraPos;
}Camera;
uniform mat4 model;
layout (location = 0) in vec3 position;
void main()
{
	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
}

#fragment
#version 430 core
void main()
{
}
