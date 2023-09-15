////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace matx
{

  class MemcpyExecutor
  {
    public:
      MemcpyExecutor(cudaStream_t stream) : stream_(stream) {}
      MemcpyExecutor(int stream) : stream_(reinterpret_cast<cudaStream_t>(stream)) {}

      template <typename Op>
      void Exec(Op &op) const noexcept {
          static_assert(Op::Rank()<0, "MemcpyExecutor can only set one tensor view to another");
      }

      /**
       * @brief Execute an operator
       *
       * @tparam Dst Destination type
       * @tparam Src Source type
       * @param dst destination tensor
       * @param src source tensor
       */
      template <typename Dst, typename Src>
      void Exec(Dst &dst, Src &src) const noexcept {

        std::vector<index_t> size;
        std::vector<index_t> srcStride;
        std::vector<index_t> dstStride;
        std::vector<index_t> copyRank;

        // Smallest dimension must be stride=1
        if (Dst::Rank() > 0 && (dst.Stride(Dst::Rank()-1) != 1 || src.Stride(Dst::Rank()-1) != 1))
        {
          size.push_back(1);
          dstStride.push_back(1);
          srcStride.push_back(1);
        }

        // Push the noncloned dimensions
        for (int dim = Dst::Rank()-1; dim >= 0; --dim)
        {
          if (dst.Size(dim) == 1) continue;
          MATX_ASSERT_STR(dst.Stride(dim) != 0, matxNotSupported, "Cannot write into a nontrivial clone");
          if (src.Stride(dim) != 0)
          {
            size.push_back(dst.Size(dim));
            dstStride.push_back(dst.Stride(dim));
            srcStride.push_back(src.Stride(dim));
          }
        }

        // Push the cloned dimensions
        for (int dim = Dst::Rank()-1; dim >= 0; --dim)
        {
          if (dst.Size(dim) == 1) continue;
          if (src.Stride(dim) == 0)
          {
            size.push_back(dst.Size(dim));
            dstStride.push_back(dst.Stride(dim));
            srcStride.push_back(src.Stride(dim));
          }
        }

        // Check if cpy0D, cpy1D, cpy2D or cpy3D can be used. Otherwise, batch this dimension.
        for (int dim = 0; dim <= size.size(); ++dim)
        {
          if (dim == 0)
          {
            copyRank.push_back(0);
          }
          else if (dim == 1 && srcStride[0])
          {
            copyRank.push_back(1);
          }
          else if (dim == 2 && srcStride[0] && srcStride[1])
          {
            copyRank.push_back(2);
          }
          else if (dim == 3
               && srcStride[0] && srcStride[1] && srcStride[2]
               && srcStride[2] % srcStride[1] == 0
               && dstStride[2] % dstStride[1] == 0)
          {
            copyRank.push_back(3);
          }
          else // batch
          {
            copyRank.push_back(4);
          }
        }

        copy<typename Dst::scalar_type>(dst.Data(), src.Data(), size, dstStride, srcStride, size.size(), copyRank);
      }

    private:
      template <typename Scalar>
      void copy(Scalar* dst,
              Scalar* src,
              std::vector<index_t>& size,
              std::vector<index_t>& dstStride,
              std::vector<index_t>& srcStride,
              index_t rank,
              std::vector<index_t>& copyRank) const noexcept {

        constexpr auto scalar_bytes = sizeof(Scalar);
        cudaError_t error = cudaSuccess;

        auto r = copyRank[rank];
        if (r == 4)
        {
          for (index_t idx = 0u; idx < size[rank-1]; ++idx)
          {
            copy(dst + idx * dstStride[rank-1], src + idx * srcStride[rank-1], size, dstStride, srcStride, rank-1, copyRank);
          }
        }
        else if (r == 3)
        {
          cudaMemcpy3DParms p = {0};
          p.dstPtr = make_cudaPitchedPtr (dst, dstStride[1] * scalar_bytes, 0, dstStride[2]/dstStride[1]);
          p.srcPtr = make_cudaPitchedPtr (src, srcStride[1] * scalar_bytes, 0, srcStride[2]/srcStride[1]);
          p.extent = {size_t(size[0] * scalar_bytes), size_t(size[1]), size_t(size[2])};
          p.kind = cudaMemcpyDefault;
          error = cudaMemcpy3DAsync(&p, stream_);
        }
        else if (r == 2)
        {
          auto dpitch = dstStride[1] * scalar_bytes;
          auto spitch = srcStride[1] * scalar_bytes;
          auto width = size[0] * scalar_bytes;
          auto height = size[1];
          error = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault, stream_);
        }
        else if (r == 1)
        {
          auto count = size[0] * scalar_bytes;
          error = cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream_);
        }
        else // (r == 0)
        {
          error = cudaMemcpyAsync(dst, src, scalar_bytes, cudaMemcpyDefault, stream_);
        }
        // std::cout << cudaGetErrorString(error) << std::endl;
      }

      cudaStream_t stream_;
  };
};
