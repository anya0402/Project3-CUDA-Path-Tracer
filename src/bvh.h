#pragma once

#include <glm/detail/type_vec.hpp>
#include <algorithm>
#include "glm/glm.hpp"
#include <string>
#include <vector>
#include "scene.h"

#include "utilities.h"
#include "sceneStructs.h"


class BVH {
public:
    BVH(std::vector<Triangle>& triangle);
    void constructBVH();
    std::vector<BVHNode> bvhNodes;
    std::vector<int> triangles_idx;
    void updateBVHBounds(int node_idx);
    void subdivideBVH(int node_idx);
    std::vector<Triangle>& mesh_triangles;
};