#pragma once

#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <vector>

#include "bvh.cuh"
#include "utils.cuh"

template <bool HasTexCoord>
struct Mesh {
 public:
  std::vector<Face<HasTexCoord>> faces;
  int32_t texture_id = -1;
};

struct Image {
 public:
  int height = 0, width = 0;
  std::string data;
};

namespace {
template <bool HasTexCoord>
void SetFace(Face<HasTexCoord> *face, int i, glm::vec3 position,
             glm::vec2 tex_coord) {
  face->position(i) = position;
  face->tex_coord(i) = tex_coord;
}
template <>
void SetFace<false>(Face<false> *face, int i, glm::vec3 position,
                    glm::vec2 tex_coord) {
  face->position(i) = position;
}

template <typename T>
glm::mat4 Mat4FromAimatrix4x4(aiMatrix4x4t<T> matrix) {
  glm::mat4 res;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) res[j][i] = matrix[i][j];
  return res;
}
};  // namespace

template <bool HasTexCoord>
struct Model {
 public:
  std::vector<Mesh<HasTexCoord>> meshes;
  std::vector<Image> textures;

  Model(const std::string &path, glm::mat4 transform) {
    const aiScene *scene = aiImportFile(
        path.c_str(), aiProcess_GlobalScale | aiProcess_CalcTangentSpace |
                          aiProcess_Triangulate);
    textures.resize(scene->mNumMaterials);
    meshes.resize(scene->mNumMeshes);
    RecursivelyInitNodes(path, scene, scene->mRootNode, transform);
    aiReleaseImport(scene);
  }

 private:
  void RecursivelyInitNodes(const std::string &root_path, const aiScene *scene,
                            aiNode *node, glm::mat4 parent_transform) {
    LOG(INFO) << "initializing node \"" << node->mName.C_Str() << "\"...";
    glm::mat4 transform =
        parent_transform * Mat4FromAimatrix4x4(node->mTransformation);
    for (int i = 0; i < node->mNumMeshes; i++) {
      int id = node->mMeshes[i];
      auto mesh = scene->mMeshes[id];
      int texture_id = mesh->mMaterialIndex;
      auto material = scene->mMaterials[texture_id];
      if (material->GetTextureCount(aiTextureType_DIFFUSE) >= 1 &&
          textures[texture_id].height == 0) {
        aiString path;
        material->GetTexture(aiTextureType_DIFFUSE, 0, &path);
        int channels;
        std::string image_path = ParentPath(ParentPath(root_path)) +
                                 "/textures/" + BaseName(path.C_Str());
        LOG(INFO) << "loading texture at: \"" << image_path << "\"";
        auto data = stbi_load(image_path.c_str(), &textures[texture_id].width,
                              &textures[texture_id].height, &channels, 3);
        textures[texture_id].data =
            std::string(data, data + (textures[texture_id].width *
                                      textures[texture_id].height * 3));
        stbi_image_free(data);
        meshes[id].texture_id = texture_id;
      }

      for (int j = 0; j < mesh->mNumFaces; j++) {
        Face<HasTexCoord> face;
        for (int k = 0; k < 3; k++) {
          int idx = mesh->mFaces[j].mIndices[k];
          auto vertex = mesh->mVertices[idx];
          glm::vec2 vec2;
          if (HasTexCoord) {
            vec2 = glm::vec2(mesh->mTextureCoords[0][idx].x,
                             mesh->mTextureCoords[0][idx].y);
          }
          auto vec4 = transform * glm::vec4(vertex.x, vertex.y, vertex.z, 1);
          auto vec3 = glm::vec3(vec4) / vec4.w;
          SetFace(&face, k, vec3, vec2);
        }
        meshes[id].faces.emplace_back(face);
      }
    }
    for (int i = 0; i < node->mNumChildren; i++) {
      RecursivelyInitNodes(root_path, scene, node->mChildren[i], transform);
    }
  }
};
