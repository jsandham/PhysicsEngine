VERTEX:

#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}Camera;

uniform mat4 model;

in vec3 position;

void main()
{
	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
}


FRAGMENT:

#version 330 core

out vec4 FragColor;

void main()
{
	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}