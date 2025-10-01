#include "bvh.h"


void BVH::constructBVH() {
    for (int i = 0; i < N; i++) {
        triangles_idx[i] = i;
    }
    memset(bvhNodes, 0, N * 2 * sizeof(BVHNode));
    int root_node_idx = 0;
    BVHNode& root = bvhNodes[root_node_idx];
    root.leftChild = -1;
    root.rightChild = -1;
    root.firstTriangleIdx = 0;
    root.triangleCount = N;
    updateBVHBounds(root_node_idx);
    subdivideBVH(root_node_idx);
}


void BVH::updateBVHBounds(int node_idx){
    BVHNode& node = bvhNodes[node_idx];
    node.aabbMin = glm::vec3(1e30f, 1e30f, 1e30f);
    node.aabbMax = glm::vec3(-1e30f, -1e30f, -1e30f);
    for (int first = node.firstTriangleIdx, i = 0; i < node.triangleCount; i++)
    {
        // triangles has info with all triangles
        Triangle leaf_triangle = mesh_triangles[triangles_idx[first + i]];
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[0]);
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[1]);
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[2]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[0]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[1]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[2]);
    }
}

void BVH::subdivideBVH(int node_idx) {
    BVHNode& node = bvhNodes[node_idx];
	glm::vec3 bbox_size = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (bbox_size[1] > bbox_size[0]) {
        axis = 1;
    }
    else if (bbox_size[2] > bbox_size[0] && bbox_size[2] > bbox_size[1]) {
        axis = 2;
	}
	float split_val = 0.5f * (node.aabbMin[axis] + node.aabbMax[axis]);

    int first_idx = node.firstTriangleIdx;
    int last_idx = first_idx + node.triangleCount - 1;
    while (first_idx <= last_idx)
    {
        if (mesh_triangles[triangles_idx[first_idx]].centroid[axis] < split_val) {
            first_idx++;
        }
        else {
			int temp = triangles_idx[last_idx];
            triangles_idx[last_idx] = triangles_idx[first_idx];
            triangles_idx[first_idx] = temp;
            last_idx--;
        }
    }

	int num_tri_left = first_idx - node.firstTriangleIdx;
	int num_tri_right = node.triangleCount - num_tri_left;
    if (num_tri_left == 0 || num_tri_right == 0) {
        return;
	}
	int left_idx = nodes_used++;
	int right_idx = nodes_used++;
	node.leftChild = left_idx;
	node.rightChild = right_idx;
    BVHNode& left_node = bvhNodes[left_idx];
	BVHNode& right_node = bvhNodes[right_idx];
	left_node.firstTriangleIdx = node.firstTriangleIdx;
	left_node.triangleCount = num_tri_left;
	right_node.firstTriangleIdx = first_idx;
	right_node.triangleCount = num_tri_right;
	node.triangleCount = 0;

    updateBVHBounds(left_idx);
    updateBVHBounds(right_idx);
    subdivideBVH(left_idx);
    subdivideBVH(right_idx);
}