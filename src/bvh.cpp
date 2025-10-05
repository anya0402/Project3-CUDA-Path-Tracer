#include "bvh.h"



BVH::BVH(std::vector<Triangle>& triangle) : mesh_triangles(triangle)
{
}

void BVH::constructBVH()
{
    int N = mesh_triangles.size();
    for (int i = 0; i < N; ++i) {
        triangles_idx.push_back(i);
    }
    int root_node_idx = 0;
    bvhNodes.push_back(BVHNode());
    bvhNodes[0].triangleCount = N;
    bvhNodes[0].firstTriangleIdx = 0;

    updateBVHBounds(root_node_idx);
    subdivideBVH(root_node_idx);
}

void BVH::updateBVHBounds(int node_idx) {
    BVHNode& node = bvhNodes[node_idx];
    node.aabbMin = glm::vec3(1e30f, 1e30f, 1e30f);
    node.aabbMax = glm::vec3(-1e30f, -1e30f, -1e30f);
    for (int first = node.firstTriangleIdx, i = 0; i < node.triangleCount; i++)
    {
        // triangles has info with all triangles
        Triangle& leaf_triangle = mesh_triangles[triangles_idx[first + i]];
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[0]);
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[1]);
        node.aabbMin = glm::min(node.aabbMin, leaf_triangle.vertices[2]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[0]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[1]);
        node.aabbMax = glm::max(node.aabbMax, leaf_triangle.vertices[2]);
    }
}

void BVH::subdivideBVH(int node_idx) {

    // ** caused MAJOR slowdown :( **
    //BVHNode& node = bvhNodes[node_idx]; 

    glm::vec3 bbox_size = (bvhNodes[node_idx].aabbMax - bvhNodes[node_idx].aabbMin);
    int axis = 0;
    if (bbox_size.y > bbox_size.x) {
        axis = 1;
    }
    if (bbox_size.z > bbox_size[axis]) {
        axis = 2;
    }
    float split_val = bvhNodes[node_idx].aabbMin[axis] + bbox_size[axis] * 0.5f;

    int first_idx = bvhNodes[node_idx].firstTriangleIdx;
    int last_idx = first_idx + bvhNodes[node_idx].triangleCount - 1;
    while (first_idx <= last_idx) {
        Triangle tri = mesh_triangles[triangles_idx[first_idx]];
        glm::vec3 centroid = (tri.vertices[0] + tri.vertices[1] + tri.vertices[2]) * 0.3333f;
        if (centroid[axis] <= split_val) {
            first_idx++;
        }
        else {
            int temp = triangles_idx[last_idx];
            triangles_idx[last_idx] = triangles_idx[first_idx];
            triangles_idx[first_idx] = temp;
            last_idx--;

        }
    }

    int num_tri_left = first_idx - bvhNodes[node_idx].firstTriangleIdx;
    int num_tri_right = bvhNodes[node_idx].triangleCount - num_tri_left;
    if (num_tri_left == 0 || num_tri_right == 0) {
        return;
    }

    bvhNodes[node_idx].leftChild = bvhNodes.size();
    bvhNodes[node_idx].rightChild = bvhNodes.size() + 1;
    bvhNodes.push_back(BVHNode());
    bvhNodes.push_back(BVHNode());
    bvhNodes[bvhNodes[node_idx].leftChild].firstTriangleIdx = bvhNodes[node_idx].firstTriangleIdx;
    bvhNodes[bvhNodes[node_idx].leftChild].triangleCount = num_tri_left;
    bvhNodes[bvhNodes[node_idx].rightChild].firstTriangleIdx = first_idx;
    bvhNodes[bvhNodes[node_idx].rightChild].triangleCount = num_tri_right;
    bvhNodes[node_idx].triangleCount = 0;

    updateBVHBounds(bvhNodes[node_idx].leftChild);
    updateBVHBounds(bvhNodes[node_idx].rightChild);
    if (bvhNodes[bvhNodes[node_idx].leftChild].triangleCount > 1) {
        subdivideBVH(bvhNodes[node_idx].leftChild);
    }
    if (bvhNodes[bvhNodes[node_idx].rightChild].triangleCount > 1) {
        subdivideBVH(bvhNodes[node_idx].rightChild);
    }
}