//#include "bvh.h"
//
//void BVH:startBVH() {
//	all_geom_indices.resize(all_geoms.size());
//	// fill indices
//	for (int i = 0; i < all_geoms.size(); i++) {
//		all_geom_indices[i] = i;
//	}
//	// clear node list
//	all_nodes.clear();
//	// construct bvh
//	root = construct_bvh(0, all_geoms.size());
//}
//
//BVHNode* BVH:contrustBVH(int start, int end) {
//	BBox bbox;
//
//	//expand bounding box
//	for (int i = start; i < end; i++) {
//		// get bbox from geom/primitive
//		bbox.expand(all_geoms[all_geom_indices[i]].bbox);
//	}
//	node.bb = bbox;
//	BVHNode* node = new BVHNode(bbox);
//
//	int num_geoms = end - start;
//	if (num_geoms <= max_leaf_size) {
//		node.left = NULL;
//		node.right = NULL;
//		node.start = start;
//		node.count = num_geoms;
//		all_nodes.push_back(node);
//	}
//	else {
//
//	}
//
//
//
//}