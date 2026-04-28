#include "dvs/scene_data.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dvs {
namespace {

constexpr int kFrameStrideFloats = 24;  // t + cam_origin(3) + cam_world(16) + light_dir(3) + intensity

std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

bool read_binary_f32(const std::filesystem::path& path, std::vector<float>* out, std::string* err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        if (err != nullptr) {
            *err = "failed to open " + path.string();
        }
        return false;
    }
    fin.seekg(0, std::ios::end);
    const std::streamoff size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        if (err != nullptr) {
            *err = "invalid f32 binary size for " + path.string();
        }
        return false;
    }
    out->resize(static_cast<std::size_t>(size / static_cast<std::streamoff>(sizeof(float))));
    fin.read(reinterpret_cast<char*>(out->data()), size);
    if (!fin) {
        if (err != nullptr) {
            *err = "failed to read " + path.string();
        }
        return false;
    }
    return true;
}

bool read_binary_u32(const std::filesystem::path& path, std::vector<std::uint32_t>* out, std::string* err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        if (err != nullptr) {
            *err = "failed to open " + path.string();
        }
        return false;
    }
    fin.seekg(0, std::ios::end);
    const std::streamoff size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0) {
        if (err != nullptr) {
            *err = "invalid u32 binary size for " + path.string();
        }
        return false;
    }
    out->resize(static_cast<std::size_t>(size / static_cast<std::streamoff>(sizeof(std::uint32_t))));
    fin.read(reinterpret_cast<char*>(out->data()), size);
    if (!fin) {
        if (err != nullptr) {
            *err = "failed to read " + path.string();
        }
        return false;
    }
    return true;
}

bool parse_metadata(
    const std::filesystem::path& path, std::unordered_map<std::string, std::string>* kv, std::string* err) {
    std::ifstream fin(path);
    if (!fin) {
        if (err != nullptr) {
            *err = "failed to open " + path.string();
        }
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (!key.empty()) {
            (*kv)[key] = value;
        }
    }
    return true;
}

bool load_mesh_data(const std::filesystem::path& dir, MeshData* out, std::string* err) {
    std::vector<float> vertices_raw;
    std::vector<float> normals_raw;
    std::vector<std::uint32_t> indices_raw;

    if (!read_binary_f32(dir / "mesh_vertices.bin", &vertices_raw, err)) {
        return false;
    }
    if (!read_binary_f32(dir / "mesh_normals.bin", &normals_raw, err)) {
        return false;
    }
    if (!read_binary_u32(dir / "mesh_indices.bin", &indices_raw, err)) {
        return false;
    }

    if (vertices_raw.size() % 3 != 0 || normals_raw.size() % 3 != 0 || indices_raw.size() % 3 != 0) {
        if (err != nullptr) {
            *err = "mesh binaries must be Nx3";
        }
        return false;
    }

    out->vertices.resize(vertices_raw.size() / 3);
    out->normals.resize(normals_raw.size() / 3);
    out->indices.resize(indices_raw.size() / 3);

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(out->vertices.size()); ++i) {
        out->vertices[i] = Vec3{
            vertices_raw[static_cast<std::size_t>(i) * 3 + 0],
            vertices_raw[static_cast<std::size_t>(i) * 3 + 1],
            vertices_raw[static_cast<std::size_t>(i) * 3 + 2],
        };
    }
    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(out->normals.size()); ++i) {
        out->normals[i] = Vec3{
            normals_raw[static_cast<std::size_t>(i) * 3 + 0],
            normals_raw[static_cast<std::size_t>(i) * 3 + 1],
            normals_raw[static_cast<std::size_t>(i) * 3 + 2],
        };
    }
    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(out->indices.size()); ++i) {
        out->indices[i] = UInt3{
            indices_raw[static_cast<std::size_t>(i) * 3 + 0],
            indices_raw[static_cast<std::size_t>(i) * 3 + 1],
            indices_raw[static_cast<std::size_t>(i) * 3 + 2],
        };
    }

    return true;
}

bool load_frames_data(
    const std::filesystem::path& dir, std::vector<FrameData>* out, std::string* err) {
    std::vector<float> raw;
    if (!read_binary_f32(dir / "frames.bin", &raw, err)) {
        return false;
    }
    if (raw.size() % kFrameStrideFloats != 0) {
        if (err != nullptr) {
            *err = "frames.bin has invalid stride";
        }
        return false;
    }

    const std::size_t frame_count = raw.size() / kFrameStrideFloats;
    out->resize(frame_count);
    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(frame_count); ++i) {
        const std::size_t o = i * kFrameStrideFloats;
        FrameData frame;
        frame.timestamp = raw[o + 0];
        frame.camera_origin = Vec3{raw[o + 1], raw[o + 2], raw[o + 3]};
        for (int j = 0; j < 16; ++j) {
            frame.camera_world.m[j] = raw[o + 4 + j];
        }
        frame.light_dir = Vec3{raw[o + 20], raw[o + 21], raw[o + 22]};
        frame.light_intensity = raw[o + 23];
        (*out)[i] = frame;
    }
    return true;
}

Vec3 triangle_centroid(const Triangle& t) {
    return (t.v0 + t.v1 + t.v2) / 3.0f;
}

AABB empty_aabb() {
    constexpr float inf = std::numeric_limits<float>::infinity();
    return AABB{
        Vec3{inf, inf, inf},
        Vec3{-inf, -inf, -inf},
    };
}

void expand_aabb(AABB* box, const Vec3& p) {
    box->min = min_vec3(box->min, p);
    box->max = max_vec3(box->max, p);
}

AABB triangle_aabb(const Triangle& t) {
    AABB box = empty_aabb();
    expand_aabb(&box, t.v0);
    expand_aabb(&box, t.v1);
    expand_aabb(&box, t.v2);
    return box;
}

}  // namespace

std::vector<Triangle> build_triangles_from_mesh(const MeshData& mesh) {
    std::vector<Triangle> tmp(mesh.indices.size());
    std::vector<std::uint8_t> valid(mesh.indices.size(), static_cast<std::uint8_t>(0));

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(mesh.indices.size()); ++i) {
        const UInt3 tri = mesh.indices[static_cast<std::size_t>(i)];
        if (tri.x >= mesh.vertices.size() || tri.y >= mesh.vertices.size() || tri.z >= mesh.vertices.size()) {
            continue;
        }
        const Vec3 v0 = mesh.vertices[static_cast<std::size_t>(tri.x)];
        const Vec3 v1 = mesh.vertices[static_cast<std::size_t>(tri.y)];
        const Vec3 v2 = mesh.vertices[static_cast<std::size_t>(tri.z)];
        const Vec3 face_n = normalize(cross(v1 - v0, v2 - v0));

        const Vec3 n0 = (tri.x < mesh.normals.size()) ? normalize(mesh.normals[static_cast<std::size_t>(tri.x)]) : face_n;
        const Vec3 n1 = (tri.y < mesh.normals.size()) ? normalize(mesh.normals[static_cast<std::size_t>(tri.y)]) : face_n;
        const Vec3 n2 = (tri.z < mesh.normals.size()) ? normalize(mesh.normals[static_cast<std::size_t>(tri.z)]) : face_n;

        tmp[static_cast<std::size_t>(i)] = Triangle{v0, v1, v2, n0, n1, n2};
        valid[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(1);
    }

    std::vector<Triangle> triangles;
    triangles.reserve(mesh.indices.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        if (valid[i] == 0) {
            continue;
        }
        triangles.push_back(tmp[i]);
    }
    return triangles;
}

void build_bvh(
    const std::vector<Triangle>& triangles,
    std::vector<BVHNode>* out_nodes,
    std::vector<Triangle>* out_ordered_triangles,
    int leaf_size) {
    out_nodes->clear();
    out_ordered_triangles->clear();
    if (triangles.empty()) {
        return;
    }

    std::vector<int> prim_indices(triangles.size());
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        prim_indices[i] = static_cast<int>(i);
    }

    struct Builder {
        const std::vector<Triangle>& tris;
        std::vector<BVHNode>* nodes;
        std::vector<Triangle>* ordered;
        std::vector<int>* prim_idx;
        int leaf_sz = 8;

        int build_node(int begin, int end) {
            const int node_idx = static_cast<int>(nodes->size());
            nodes->push_back(BVHNode{});
            BVHNode node;

            AABB bounds = empty_aabb();
            AABB centroid_bounds = empty_aabb();
            for (int i = begin; i < end; ++i) {
                const Triangle& tri = tris[static_cast<std::size_t>((*prim_idx)[i])];
                const AABB tri_box = triangle_aabb(tri);
                expand_aabb(&bounds, tri_box.min);
                expand_aabb(&bounds, tri_box.max);
                expand_aabb(&centroid_bounds, triangle_centroid(tri));
            }

            const int count = end - begin;
            node.box = bounds;
            if (count <= leaf_sz) {
                node.start = static_cast<int>(ordered->size());
                node.count = count;
                node.left = -1;
                node.right = -1;
                for (int i = begin; i < end; ++i) {
                    ordered->push_back(tris[static_cast<std::size_t>((*prim_idx)[i])]);
                }
                (*nodes)[node_idx] = node;
                return node_idx;
            }

            const Vec3 ext = centroid_bounds.max - centroid_bounds.min;
            int axis = 0;
            if (ext.y > ext.x && ext.y >= ext.z) {
                axis = 1;
            } else if (ext.z > ext.x && ext.z > ext.y) {
                axis = 2;
            }

            const int mid = begin + count / 2;
            std::nth_element(
                prim_idx->begin() + begin,
                prim_idx->begin() + mid,
                prim_idx->begin() + end,
                [&](int a, int b) {
                    return triangle_centroid(tris[static_cast<std::size_t>(a)])[axis] <
                           triangle_centroid(tris[static_cast<std::size_t>(b)])[axis];
                });

            node.left = build_node(begin, mid);
            node.right = build_node(mid, end);
            node.start = 0;
            node.count = 0;
            (*nodes)[node_idx] = node;
            return node_idx;
        }
    };

    Builder builder{
        triangles,
        out_nodes,
        out_ordered_triangles,
        &prim_indices,
        std::max(1, leaf_size),
    };
    builder.build_node(0, static_cast<int>(triangles.size()));
}

bool load_scene_bundle(
    const std::filesystem::path& bundle_dir,
    SceneBundle* out_bundle,
    std::string* error_message) {
    std::unordered_map<std::string, std::string> meta;
    if (!parse_metadata(bundle_dir / "metadata.txt", &meta, error_message)) {
        return false;
    }

    if (meta.find("width") == meta.end() || meta.find("height") == meta.end()) {
        if (error_message != nullptr) {
            *error_message = "metadata.txt missing width/height";
        }
        return false;
    }
    std::cout << "Loading scene bundle from " << bundle_dir << " with resolution " << meta["width"] << "x" << meta["height"] << std::endl;

    MeshData mesh;
    if (!load_mesh_data(bundle_dir, &mesh, error_message)) {
        return false;
    }

    std::cout << "Loaded mesh with " << mesh.vertices.size() << " vertices, "
              << mesh.normals.size() << " normals, and " << mesh.indices.size()
              << " triangles" << std::endl;
    
    std::vector<FrameData> frames;
    if (!load_frames_data(bundle_dir, &frames, error_message)) {
        return false;
    }
    if (frames.empty()) {
        if (error_message != nullptr) {
            *error_message = "frames.bin is empty";
        }
        return false;
    }

    SceneBundle bundle;
    bundle.width = std::stoi(meta["width"]);
    bundle.height = std::stoi(meta["height"]);
    bundle.fps = (meta.find("fps") != meta.end()) ? std::stof(meta["fps"]) : 0.0f;
    bundle.frames = std::move(frames);
    bundle.fx = std::stof(meta["fx"]);
    bundle.fy = std::stof(meta["fy"]);
    bundle.cx = std::stof(meta["cx"]);
    bundle.cy = std::stof(meta["cy"]);

    bundle.bvh_triangles = build_triangles_from_mesh(mesh);
    bundle.bvh_nodes.clear();

    if (bundle.bvh_triangles.empty()) {
        if (error_message != nullptr) {
            *error_message = "mesh is empty after triangulation";
        }
        return false;
    }

    *out_bundle = std::move(bundle);
    return true;
}

}  // namespace dvs
