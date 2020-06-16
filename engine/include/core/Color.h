#ifndef __COLOR_H__
#define __COLOR_H__

namespace PhysicsEngine
{
	class Color
	{
		public:
			static Color white;
			static Color black;
			static Color red;
			static Color green;
			static Color blue;
			static Color yellow;

			float r;
			float g;
			float b;
			float a;

		public:
			Color();
			Color(float r, float g, float b, float a);
			~Color();
	};

	class Color32
	{
	public:
		static Color32 white;
		static Color32 black;
		static Color32 red;
		static Color32 green;
		static Color32 blue;
		static Color32 yellow;

		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;

	public:
		Color32();
		Color32(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
		~Color32();
	};
}

#endif