layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}Camera;
in vec3 position;
in vec3 normal;
in vec2 texCoord;
out vec3 FragPos;
out vec3 Normal;
uniform mat4 model;
void main()
{
    vec4 viewPos = Camera.view * model * vec4(position, 1.0);
    FragPos = viewPos.xyz;
    mat3 normalMatrix = transpose(inverse(mat3(Camera.view * model)));
    Normal = normalMatrix * normal;
    gl_Position = Camera.projection * viewPos;
}
