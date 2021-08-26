in vec3 Normal;
out vec4 FragColor;
void main()
{
	FragColor = vec4(Normal.xyz, 1.0f);
}
