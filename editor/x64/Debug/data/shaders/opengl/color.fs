struct Material
{
   vec4 color;
};
uniform Material material;
out vec4 FragColor;
void main()
{
#if defined(DIRECTIONALLIGHT)
#if defined(SOFTSHADOWS) || defined(HARDSHADOWS)
	FragColor = material.color;
#endif
#else
	FragColor = material.color;
#endif
}