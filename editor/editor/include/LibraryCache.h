#ifndef __LIBRARY_CACHE_H__
#define __LIBRARY_CACHE_H__

#include <string>
#include <map>

namespace PhysicsEditor
{
	typedef struct FileInfo
	{
		std::string filePath;
		std::string fileExtension;
		std::string createTime;
		std::string accessTime;
		std::string writeTime;
	}FileInfo;

	class LibraryCache
	{
		private:
			std::map<std::string, FileInfo> filePathToFileInfo;

			friend class iterator;

		public:
			LibraryCache();
			~LibraryCache();

			void add(std::string filePath, FileInfo fileInfo);
			void remove(std::string filePath);
			void clear();
			bool load(std::string libraryCachePath);
			bool save(std::string libraryCachePath);
			bool contains(std::string filePath) const;
			bool isOutOfDate(std::string filePath, std::string createTime, std::string writeTime) const;

			//friend class const_iterator;

			class iterator
			{
				private:
					std::map<std::string, FileInfo>::iterator it;

					friend class LibraryCache;

				public:
					iterator();
					~iterator();

					bool operator==(const iterator& other) const;
					bool operator!=(const iterator& other) const;

					iterator& operator++();
					iterator operator++(int);
					iterator& operator--();
					iterator operator--(int);

					std::pair<const std::string, FileInfo>& operator*() const;
					std::pair<const std::string, FileInfo>* operator->() const;
			};

			//iterator begin();
			//const_iterator begin() const;

			//iterator end();
			//const_iterator end() const;

			iterator begin()
			{
				iterator iter;
				iter.it = filePathToFileInfo.begin();
				return iter;
			}

			iterator end()
			{
				iterator iter;
				iter.it = filePathToFileInfo.end();
				return iter;
			}

			iterator find(const std::string filePath)
			{
				iterator iter;
				for (iterator it = begin(); it != end(); it++) {
					if (it->first == filePath) {
						return it;
					}
				}

				return iter;
			}
	};





	/*class iterator
	{
		private:
			std::map<std::string, FileInfo>::iterator it;

			friend class LibraryCache;

		public:
			iterator();
			~iterator();

			bool operator==(const iterator& other) const;
			bool operator!=(const iterator& other) const;

			iterator& operator++();
			iterator& operator++(int);
			iterator& operator--();
			iterator& operator--(int);

			std::pair<std::string, FileInfo> operator*() const;
			std::pair<std::string, FileInfo>* operator->() const;
	};*/

	/*class const_iterator
	{
		private:
			const std::map<std::string, FileInfo>::const_iterator it;

			friend class LibraryCache;

		public:
			const_iterator();
			~const_iterator();

			bool operator==(const const_iterator& other) const;
			bool operator!=(const const_iterator& other) const;

			const_iterator& operator++();
			const_iterator& operator++(int);

			std::pair<std::string, FileInfo> operator*() const;
			std::pair<std::string, FileInfo>* operator->() const;
	};*/
}

#endif
