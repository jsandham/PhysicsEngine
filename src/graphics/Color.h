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

			unsigned char r;
			unsigned char g;
			unsigned char b;
			unsigned char a;

		public:
			Color();
			Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
			~Color();
	};
}

#endif