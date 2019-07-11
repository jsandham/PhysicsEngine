#ifndef __SOLID_H__
#define __SOLID_H__

#include <vector>

#include "Component.h"

namespace PhysicsEngine
{
	class Solid : public Component
	{
		public:
		    float c;                //specific heat coefficient                         
		    float rho;              //density                            
		    float Q;                //internal heat generation   
		    float k;                //thermal conductivity coefficient

			int dim;                //dimension of mesh (1, 2, or 3) 
		    int ng;                 //number of element groups
		    int n;                  //total number of nodes                      
		    int nte;                //total number of elements (Nte=Ne+Ne_b)       
		    int ne;                 //number of interior elements                
		    int ne_b;               //number of boundary elements                                 
		    int npe;                //number of points per interior element      
		    int npe_b;              //number of points per boundary element      
		    int type;               //interior element type                      
		    int type_b;             //boundary element type    

			std::vector<float> vertices;
			std::vector<int> connect;
			std::vector<int> bconnect;
			std::vector<int> groups;

			// Buffer vertexVBO;
			// Buffer normalVBO;
			// VertexArrayObject solidVAO;

		public:
			Solid();
			Solid(std::vector<char> data);
			~Solid();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);
	};
}

#endif