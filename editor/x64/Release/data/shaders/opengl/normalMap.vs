layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}Camera;
uniform mat4 model;
in vec3 position;
in vec3 normal;
out vec3 Normal;
void main()
{
	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);
   Normal = normal;
}
