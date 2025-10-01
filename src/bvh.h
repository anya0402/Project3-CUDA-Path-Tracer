#pragma once

#include <glm/detail/type_vec.hpp>
#include <algorithm>
#include "glm/glm.hpp"
#include <string>
#include <vector>
#include "scene.h"

struct aabb {

};





class BVH {
public:
	BVH(const std::vector<Triangle>& tris) : mesh_triangles(tris) {
		N = static_cast<int>(mesh_triangles.size());
		triangles_idx.resize(N);
		bvhNodes = nullptr;
		nodes_used = 1;
	}

	void constructBVH();
	BVHNode* bvhNodes;
	int nodes_used;

private:
	void updateBVHBounds(int node_idx);
	void subdivideBVH(int node_idx);
	const std::vector<Triangle>& mesh_triangles;
	std::vector<int> triangles_idx;
	int N;
};
