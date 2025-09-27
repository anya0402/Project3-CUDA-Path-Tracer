//#include <glm/detail/type_vec.hpp>
//#include <algorithm>
//
//struct Bbox {
//	glm::vec3 max;
//	glm::vec3 min;
//	glm::vec3 span;
//
//	BBox() {
//		max = glm::vec3(0, 0, 0);
//		min = glm::vec3(0, 0, 0);
//		span = max - min;
//	}
//
//	void expand(const BBox& bbox) {
//		min.x = std::min(min.x, bbox.min.x);
//		min.y = std::min(min.y, bbox.min.y);
//		min.z = std::min(min.z, bbox.min.z);
//		max.x = std::max(max.x, bbox.max.x);
//		max.y = std::max(max.y, bbox.max.y);
//		max.z = std::max(max.z, bbox.max.z);
//		span = max - min;
//	}
//
//	void expand(const glm::vec3 p) {
//		min.x = std::min(min.x, p.x);
//		min.y = std::min(min.y, p.y);
//		min.z = std::min(min.z, p.z);
//		max.x = std::max(max.x, p.x);
//		max.y = std::max(max.y, p.y);
//		max.z = std::max(max.z, p.z);
//		span = max - min;
//	}
//
//	glm::vec3 centroid() const {
//		return (min + max) / 2;
//	}
//
//};
//
//struct BVHNode {
//	BBox bb;
//	int left;
//	int right;
//	int start;
//	int count;
//
//	BVHNode(BBox bb) : bb(bb), left(NULL), right(NULL), start(NULL), count(NULL) {}
//};
//
//
//
////struct BVHAccel { // keep in translation-unit only
////	//BBox bb;
////	bool is_leaf;
////	int left_idx;
////	int right_idx;
////	int start; // start index into prim_indices_host
////	int range;
////};
//
//
//class BVH {
//public:
//	BVH();
//	~BVH();
//	void startBVH();
//	void BVHtoDevice();
//	void freeBVH();
//	BVHNode* bvh_nodes;
//	int* geom_indices;
//	glm::vector<BVHNode> all_nodes;
//	glm::vector<int> all_geom_indices;
//
//private:
//	BVHNode* construct_bvh(int start, int end);
//	glm::vector<Geom>* geoms;
//	int max_leaf_size;
//};
