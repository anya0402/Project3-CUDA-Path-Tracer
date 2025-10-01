#pragma once

#include "sceneStructs.h"
#include <vector>
#include "bvh.h"


class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromObj(const std::string& pathName, Geom& mesh);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
	std::vector<Texture> textures;
    std::vector<Texture> textures_norms;
    RenderState state;
};
