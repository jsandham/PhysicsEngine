#ifndef __GMESH_H__
#define __GMESH_H__

#include<vector>

#include "Guid.h"
#include "Asset.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct GMeshHeader
	{
		Guid gmeshId;
		int dim;
		int ng;
	    int n;
	    int nte;
	    int ne;
	    int ne_b;
	    int npe;
	    int npe_b;
	    int type;
	    int type_b;
		size_t verticesSize;
		size_t connectSize;
		size_t bconnectSize;
		size_t groupsSize;
	};
#pragma pack(pop)

	class GMesh : public Asset
	{
		public:
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

		public:
			GMesh();
			GMesh(std::vector<char> data);
			~GMesh();

			void* operator new(size_t size);
			void operator delete(void*);
	};
}

#endif