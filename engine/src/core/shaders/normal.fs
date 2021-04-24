STRINGIFY(
uniform int wireframe;
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;
void main(void)
{
	FragColor = vec4(wireframe * Normal, 1.0f);
}
)