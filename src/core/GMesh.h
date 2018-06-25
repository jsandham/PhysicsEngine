#ifndef __GMESH_H__
#define __GMESH_H__

#include<vector>

namespace PhysicsEngine
{
	class GMesh
	{
		private:
			std::vector<float> vertices;
			std::vector<int> connect;
			std::vector<int> bconnect;	
			std::vector<int> groups;

		public:
			GMesh();
			~GMesh();

			std::vector<float>& getVertices();
			std::vector<int>& getConnect();
			std::vector<int>& getBConnect();
			std::vector<int>& getGroups();

			void setVertices(std::vector<float> &vertices);
			void setConnect(std::vector<int> &connect);
			void setBConnect(std::vector<int> &bconnect);
			void setGroups(std::vector<int> &groups);
	};
}

#endif