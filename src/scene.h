#pragma once

#include "sceneStructs.h"
#include "bvh.h"
#include <vector>
#include <iostream>

class BVH;

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromObj(const std::string& pathName, Geom& mesh);
    int Scene::loadTexture(const std::string pathName, std::string textName);
    int Scene::loadBumpMap(const std::string pathName, const std::string textName);
    int Scene::loadEnvMap(const std::string pathName);

public:
    Scene(std::string filename);
    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Triangle> triangles;
	std::vector<int> triangles_idx;
	std::vector<Texture> textures;
    std::vector<Texture> normals;
    std::vector<Texture> env_maps;
    RenderState state;
    //std::unique_ptr<BVH> bvh;
    std::vector<BVHNode> bvhNodes;

};
