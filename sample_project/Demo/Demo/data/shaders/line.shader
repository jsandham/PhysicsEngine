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
out float z;

void main()
{
	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
	z = gl_Position.z;
}


FRAGMENT:

#version 330 core

in float z;
out vec4 FragColor;

void main()
{
	FragColor = vec4(z / 20.0, 0.0f, 0.0f, 1.0f);
}