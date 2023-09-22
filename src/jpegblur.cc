// Copyright (C) 2022 David Miguel Susano Pinto <pinto@robots.ox.ac.uk>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// This program was heavily inspired by libjpeg's jpegtran program
// which has the following license:
//
// jpegtran.c
//
// This file was part of the Independent JPEG Group's software:
// Copyright (C) 1995-2019, Thomas G. Lane, Guido Vollbeding.
// libjpeg-turbo Modifications:
// Copyright (C) 2010, 2014, 2017, 2019-2022, D. R. Commander.
// For conditions of distribution and use, see the accompanying README.ijg
// file.
//
// This file contains a command-line user interface for JPEG transcoding.
// It is very similar to cjpeg.c, and partly to djpeg.c, but provides
// lossless transcoding between different JPEG file formats.  It also
// provides some lossless and sort-of-lossless transformations of JPEG data.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <getopt.h>
#include <jpeglib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>


// We could adjust the size of individual Mat elements when reading
// the pixel values but I'm not sure what we would need to do in terms
// of scaling them.  Also, this does not matter when doing pixelation
// only.
static_assert(sizeof(JSAMPLE) == 1,
              "libjpeg not built with BITS_IN_JSAMPLE == 8");


void
print_size(const std::string& name, const cv::Mat& m)
{
  if (m.dims > 2)
    std::cerr << name << " size is :" << m.size[0] << "x" << m.size[1] << "x" << m.size[2] << "\n";
  else
    std::cerr << name << " size is (rows x cols):" << m.rows << "x" << m.cols << "\n";
 }


class BoundingBox {
public:
  const int x0;
  const int x1;
  const int y0;
  const int y1;

  BoundingBox(const int x0, const int x1, const int y0, const int y1)
     : x0{x0}, x1{x1}, y0{y0}, y1{y1}
  {}

  int
  width() const
  {
    return x1 - x0;
  }

  int
  height() const
  {
    return y1 - y0;
  }

  void
  print() const
  {
    std::cerr << "[" << x0<< ", " << y0 << ", " << x1 <<", " << y1 << "]\n";
  }

  static BoundingBox
  from_cmdline_arg(const std::string& arg)
  {
    int x0, x1, y0, y1;
    std::stringstream argss(arg);
    std::string sep("   ");
    if (! (argss >> x0 >> sep[0]
           >> x1 >> sep[1]
           >> y0 >> sep[2]
           >> y1)
        || sep != ",,,"
        || ! argss.eof())
      throw std::invalid_argument("failed to parse BB from '" + arg + "'");
    return BoundingBox{x0, x1, y0, y1};
  }

  // New BB expanded to include all MCUs under the original BB.
  BoundingBox
  expanded_to_MCU() const
  {
    const int x_x0 = x0 /8 *8;
    const int x_x1 = (x1 + (8 -1)) /8 *8;
    const int x_y0 = y0 /8 *8;
    const int x_y1 = (y1 + (8 -1)) /8 *8;
    return BoundingBox{x_x0, x_x1, x_y0, x_y1};
  }

  // New BB expanded by l on all directions.
  BoundingBox
  expand_by(const int l) const
  {
    return BoundingBox{x0 -l, x1 +l, y0 -l, y1 +l};
  }

  BoundingBox
  limit_for_size(const int ncols, const int nrows) const
  {
    return BoundingBox{
      (x0 < 0) ? 0 : x0,
      (x1 < ncols) ? x1 : ncols -1,
      (y0 < 0) ? 0 : y0,
      (y1 < nrows) ? y1 : nrows -1,
    };
  }

  cv::Mat
  to_mask_for(const cv::Mat& img) const
  {
    const std::vector<int> tmp_shape {img.rows, img.cols, img.channels()};
    cv::Mat mask = cv::Mat::zeros(3, tmp_shape.data(), CV_64F);
    mask = mask.reshape(img.channels(), 2, tmp_shape.data());
    mask(cv::Range{y0, y1+1}, cv::Range{x0, x1+1}) = cv::Scalar::all(1.0);
    return mask;
  }

};


class JPEGDecompressor {
private:
  struct jpeg_error_mgr jerr;

  // FIXME: We can't call start_decompress if we only access the
  // coefficients, and we can't call finish_decompression (in the
  // destructor) if we didn't call start_decompression.  We should
  // organize things better.
  bool started_decompress = false;

  // Setup decompression object to save all markers in memory.
  void
  save_markers()
  {
    jpeg_save_markers(&info, JPEG_COM, 0xFFFF);
    for (int m = 0; m < 16; m++)
      jpeg_save_markers(&info, JPEG_APP0 + m, 0xFFFF);
  }

public:
  struct jpeg_decompress_struct info;

  JPEGDecompressor(std::FILE *file)
  {
    info.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&info);
    jpeg_stdio_src(&info, file);

    save_markers();
    jpeg_read_header(&info, TRUE);

    // We use buffered-image mode because we want to keep the
    // full-image coefficient array in memory.  If not, coefficients
    // are discarded as we scan each line of the image.
  }

  bool
  maybe_has_jfif_thumbnail()
  {
    // Being a JFIF version 1.02, this file might have a thumbnail.
    // Even if it has no thumbnail in the JFIF marker, it might still
    // have a thumbnail on some other marker.
    return info.JFIF_minor_version > 1;
  }

  bool
  is_progressive()
  {
    return info.progressive_mode;
  }

  JBLOCKARRAY
  get_coeff_for_row(const int component_idx, const int row_idx)
  {
    jvirt_barray_ptr *coeff_arrays = jpeg_read_coefficients(&info);
    jpeg_component_info comp_info = info.comp_info[component_idx];
    // Should be possible to specify the number of rows in
    // access_virt_barray but I keep getting "Bogus virtual array
    // access".
    //    std::cerr
    return info.mem->access_virt_barray((j_common_ptr)&info,
                                        coeff_arrays[component_idx],
                                        row_idx, comp_info.v_samp_factor, TRUE);
  }

  cv::Mat
  to_image()
  {
    // We are not dealing with progressive images and only use
    // buffered-image to keep all coefficients in memory.  So we can
    // just specify scan=1 and assume we are dealing with the full
    // resolution image/scan.
    jpeg_start_decompress(&info);
    started_decompress = true;
    jpeg_start_output(&info, 1);

    // Should be possible to read the image straight into a 3 channel
    // 2D matrix but I couldn't figure out the right incantations so
    // it's created as 1 channel 3D array and reshaped at the end.
    cv::Mat rgb {{static_cast<int>(info.image_height),
                  static_cast<int>(info.image_width),
                  static_cast<int>(info.num_components)},
                 CV_8UC1};

    const int row_stride = info.image_width * info.num_components;
    JSAMPARRAY row_buffer = info.mem->alloc_sarray((j_common_ptr)&info,
                                                   JPOOL_IMAGE, row_stride, 1);

    int line_idx = 0;
    while (info.output_scanline < info.output_height) {
      jpeg_read_scanlines(&info, row_buffer, 1);
      for (int i = 0; i < row_stride; i++)
        rgb.data[line_idx*row_stride + i] = row_buffer[0][i];

      ++line_idx;
    }

    jpeg_finish_output(&info);

    return rgb.reshape(info.num_components, 2, rgb.size.p);
  }

  ~JPEGDecompressor()
  {
    if (started_decompress)
      jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
  }
};


class JPEGCompressor {
private:
  struct jpeg_error_mgr jerr;

public:
  struct jpeg_compress_struct info;

  JPEGCompressor(std::FILE *file)
  {
    info.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&info);
    jpeg_stdio_dest(&info, file);
  }

  void
  copy_critical_parameters_from(JPEGDecompressor& src)
  {
    jpeg_copy_critical_parameters(&src.info, &info);
  }

  // Copy markers saved in the given source object to the destination object.
  //
  // This should be called just after jpeg_start_compress() or
  // jpeg_write_coefficients().  Note that those routines will have
  // written the SOI, and also the JFIF APP0 or Adobe APP14 markers if
  // selected so we need to skip those.
  void
  copy_markers_from(const JPEGDecompressor& src)
  {
    for (jpeg_saved_marker_ptr marker = src.info.marker_list; marker != NULL;
         marker = marker->next) {
      if (info.write_JFIF_header &&
          marker->marker == JPEG_APP0 &&
          marker->data_length >= 5 &&
          marker->data[0] == 0x4A &&
          marker->data[1] == 0x46 &&
          marker->data[2] == 0x49 &&
          marker->data[3] == 0x46 &&
          marker->data[4] == 0)
        continue;                 // reject duplicate JFIF
      if (info.write_Adobe_marker &&
          marker->marker == JPEG_APP0 + 14 &&
          marker->data_length >= 5 &&
          marker->data[0] == 0x41 &&
          marker->data[1] == 0x64 &&
          marker->data[2] == 0x6F &&
          marker->data[3] == 0x62 &&
          marker->data[4] == 0x65)
        continue;                 // reject duplicate Adobe
      jpeg_write_marker(&info, marker->marker, marker->data,
                        marker->data_length);
    }
  }

  void
  copy_coefficients_from(JPEGDecompressor& src)
  {
    jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(&src.info);
    jpeg_write_coefficients(&info, src_coef_arrays);
  }

  ~JPEGCompressor()
  {
    jpeg_finish_compress(&info);
    jpeg_destroy_compress(&info);
  }
};


// FIXME: this should be done some other way.
class JPEGFileDecompressor : public JPEGDecompressor {
private:
  std::string fpath;

public:
  JPEGFileDecompressor(const std::string& fpath)
    : fpath{fpath}, JPEGDecompressor(std::fopen(fpath.c_str(), "rb"))
  {}

  static JPEGFileDecompressor
  from_image(const cv::Mat& img, JPEGDecompressor& src)
  {
    // FIXME: we shouldn't use tmpnam
    std::string fpath {std::tmpnam(nullptr)};
    FILE* outfp = std::fopen(fpath.c_str(), "wb");

    // This is all on its own block so that ~JPEGCompressor is called
    // at the end and we can then fclose.
    {
      JPEGCompressor dst {outfp};
      dst.copy_critical_parameters_from(src);
      dst.info.in_color_space = src.info.out_color_space;
      jpeg_start_compress(&dst.info, TRUE);

      JSAMPROW row_pointer[1];
      const int row_stride = img.cols * img.channels();
      while (dst.info.next_scanline < dst.info.image_height) {
        row_pointer[0] = &img.data[dst.info.next_scanline * row_stride];
        jpeg_write_scanlines(&dst.info, row_pointer, 1);
      }
    }
    std::fclose(outfp);

    return JPEGFileDecompressor{fpath};
  }

  ~JPEGFileDecompressor()
  {
    std::remove(fpath.c_str());
  }
};



// Create composite image by blending images using a transparency mask.
cv::Mat
composite_image(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask)
{
  // Mblurred * Iblurred + (1 - Mblurred) * I
  //    <==>
  // img2 * mask + (1 - mask) * img1
  //
  // TODO: there's gotta be a cleaner way to do this.  Check
  // https://stackoverflow.com/questions/36216702/combining-2-images-with-transparent-mask-in-opencv
  cv::Mat weighted_img2;
  cv::multiply(mask, img2, weighted_img2,
               1.0, img2.type());

  cv::Mat weighted_img1;
  cv::multiply(cv::Scalar::all(1.0) - mask, img1, weighted_img1,
               1.0, img1.type());

  cv::Mat composite;
  cv::add(weighted_img2, weighted_img1, composite);

  return composite;
}


cv::Mat
blur_region(const cv::Mat& img, const BoundingBox& bb)
{
  const double sigma = 0.1 * std::max(bb.width(), bb.height());

  // Gaussian kernel truncated at 10 sigma.  That's the value used for
  // groundtruth in
  // https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
  // Maybe we could use their extended box approach, same that is done
  // in Pillow, since our images are relatively large and some of our
  // regions (and therefore sigmas) are too.
  int length = sigma * 10;
  if (length % 2 == 0)  // OpenCV requires kernel length to be odd
    ++length;

  cv::Mat img_blurred;
  cv::GaussianBlur(img, img_blurred, cv::Size(length, length), sigma, sigma,
                   cv::BorderTypes::BORDER_REPLICATE);

  const int xt = std::max(bb.width() /10, bb.height() /10);
  const BoundingBox xl_bb = bb.expand_by(xt).limit_for_size(img.cols, img.rows);

  cv::Mat mask = xl_bb.to_mask_for(img);
  cv::Mat mask_blurred;
  cv::GaussianBlur(mask, mask_blurred, cv::Size(length, length), sigma, sigma,
                   cv::BorderTypes::BORDER_REPLICATE);

  return composite_image(img, img_blurred, mask_blurred);
}


// Similar approach described in https://arxiv.org/abs/2103.06191 (see
// Appendix B and Figure C).  Their implementation is at
// https://github.com/princetonvisualai/imagenet-face-obfuscation/blob/main/experiments/blurring.py
// and we follow it through Pillow's:
//
//   1. src/PIL/ImageFilter.py (GaussianBlur)
//   2. src/_imaging.c (_gaussian_blur)
//   3. src/libImaging/BoxBlur.c (ImagingGaussianBlur)
//
// There's two main changes:
//
//   1. we blur one region at a time instead of only once.  We do that
//      to cover the case where we have in the same image a very small
//      region and a very large region.  If we don't, then we would
//      blur with a very high sigma a very small region.
//   2. they use a gaussian blur approximation which is much faster
//      and seems to blur a lot more.
cv::Mat
blur_image(const cv::Mat& src, const std::vector<BoundingBox>& bounding_boxes)
{
  // Gaussian blur only handles 2D arrays so merge the channels and
  // split them at the end.
  cv::Mat img = src.clone();
  for (auto bb : bounding_boxes)
    img = blur_region(img, bb);

  return img;
}


// This is equivalent to cv::reduce(src, dst, 2, REDUCE_MAX) if
// cv::reduce could reduce across the 3rd dimension.
cv::Mat
reduceMax3D(const cv::Mat& src)
{
  // Don't catch dims<3 because in that case we do nothing anyway.
  if (src.dims > 3)
    throw std::invalid_argument {"SRC has more than 3 dimensions"};

  const std::vector<int> tmpshape {src.size[0] * src.size[1], src.size[2]};
  cv::Mat dst = src.reshape(1, 2, tmpshape.data());
  cv::reduce(dst, dst, 1, cv::ReduceTypes::REDUCE_MAX);
  dst = dst.reshape(1, src.size[0]);

  assert (dst.size[0] == src.size[0] && dst.size[1] == src.size[1]);
  return dst;
}


cv::Mat
reduceMaxMCU(const cv::Mat& src, std::pair<int, int> sz_in_blks)
{
  if (src.dims > 2)
    throw std::invalid_argument {"SRC has more than 2 dimensions"};

  int downsampled_blk_height = DCTSIZE;
  while (sz_in_blks.first * downsampled_blk_height < src.rows)
    downsampled_blk_height = downsampled_blk_height * 2;

  int downsampled_blk_width = DCTSIZE;
  while (sz_in_blks.second * downsampled_blk_width < src.cols)
    downsampled_blk_width = downsampled_blk_width * 2;

  const int MCUrows = ((src.rows + downsampled_blk_height -1)
                       / downsampled_blk_height);
  const int MCUcols = ((src.cols + downsampled_blk_width -1)
                       / downsampled_blk_width);

  cv::Mat dst;
  cv::copyMakeBorder(src, dst,
                     0, (MCUrows * downsampled_blk_height) - src.rows,
                     0, (MCUcols * downsampled_blk_width) - src.cols,
                     cv::BorderTypes::BORDER_CONSTANT,
                     cv::Scalar(false));

  dst = dst.reshape(1, MCUcols * MCUrows * downsampled_blk_height);
  cv::reduce(dst, dst, 1, cv::ReduceTypes::REDUCE_MAX);
  dst = dst.reshape(1, MCUrows * downsampled_blk_height);

  cv::transpose(dst, dst);
  dst = dst.reshape(1, MCUrows * MCUcols);
  cv::reduce(dst, dst, 1, cv::ReduceTypes::REDUCE_MAX);
  dst = dst.reshape(1, MCUcols);
  cv::transpose(dst, dst);

  assert (dst.cols == MCUcols && dst.rows == MCUrows);
  return dst;
}


void
merge(JPEGDecompressor& src, const cv::Mat& src_img,
      JPEGDecompressor& mod, const cv::Mat& mod_img)
{
  // Different image components may have different sampling values,
  // i.e., a single MCU may cover a different number of blocks in
  // different components so we need to compute different masks for
  // each component.  However, most channels use the same sampling so
  // instead of recreating the mask for each component, we use a map
  // and create it only once for each sampling used.
  //
  // Note that sampling notation used in jpeglib uses sampling factors
  // instead of the common J:a:b notation.  See
  // https://zpl.fi/chroma-subsampling-and-jpeg-sampling-factors/ for
  // an explanation.

  std::map<std::pair<int, int>, cv::Mat> sz_in_blk_to_mask;
  cv::Mat flat_mask = reduceMax3D(src_img != mod_img);
  for (int comp_idx = 0; comp_idx < src.info.num_components; comp_idx++) {
    jpeg_component_info comp_info = src.info.comp_info[comp_idx];
    std::pair<int, int> sz_in_blk {comp_info.height_in_blocks,
                                   comp_info.width_in_blocks};
    if (sz_in_blk_to_mask.find(sz_in_blk) == sz_in_blk_to_mask.end())
      sz_in_blk_to_mask[sz_in_blk] = reduceMaxMCU(flat_mask, sz_in_blk);
  }

  for (int comp_idx = 0; comp_idx < src.info.num_components; comp_idx++) {
    jpeg_component_info comp_info = src.info.comp_info[comp_idx];
    std::pair<int, int> sz_in_blk {comp_info.height_in_blocks,
                                   comp_info.width_in_blocks};
    cv::Mat blk_mask = sz_in_blk_to_mask[sz_in_blk];

    // We can't loop over each row directly.  Each time we get the
    // coefficients, we get rows equal to the number of vertical
    // sampling factor.  We then loop over each of the row acquired.
    for (int blk_y = 0;
         blk_y < comp_info.height_in_blocks;
         blk_y += comp_info.v_samp_factor) {
      JBLOCKARRAY src_buf = src.get_coeff_for_row(comp_idx, blk_y);
      JBLOCKARRAY mod_buf = mod.get_coeff_for_row(comp_idx, blk_y);
      for (int offset_y = 0;
           offset_y < comp_info.v_samp_factor;
           offset_y++) {

        // For each true element in the mask, copy all the
        // coefficients to the related DCT block.
        bool* mask_ptr = blk_mask.ptr<bool>(blk_y + offset_y);
        for (int blk_x = 0; blk_x < comp_info.width_in_blocks; blk_x++)
          if (mask_ptr[blk_x])
            for (int k = 0; k < DCTSIZE2; k++)
              src_buf[offset_y][blk_x][k] = mod_buf[offset_y][blk_x][k];

      }
    }
  }
}


// TODO: this function is dead code but I want to clean it up and make
// it an option.  It's much faster than bluring and does not require
// opencv.
void
pixelate_regions(const JPEGDecompressor& src,
                 jvirt_barray_ptr *src_coeffs_array,
                 const BoundingBox& bb)
{
  // TODO: this needs to be expanded before
  const BoundingBox mcu_bb = bb.expanded_to_MCU();

  // We are doing the indexing in MCU coordinates and not in pixels
  // (one MCU corresponds to 8x8 pixels).
  const int start_col = mcu_bb.x0 / 8;
  const int end_col = mcu_bb.x1 /8;
  const int ncols = end_col - start_col;

  const int start_row = mcu_bb.y0 / 8;
  const int end_row = mcu_bb.y1 / 8;
  const int nrows = end_row - start_row;

  // In addition of "destroying" each MCU information to only use the
  // first coefficient, we also join adjacent MCUs so that each region
  // is at most 8 rectangles accross any of the axis.  Examples:
  //
  //   * a region of 8x64 pixels is a region of 1x8 MCU and there are
  //     no MCUs merged, output will look like 1x8 blocks, each of
  //     them 8x8 pixels.
  //
  //   * a region of 8x128 pixels is a region of 1x16 MCU and MCUs
  //     will merged in pairs, the output beking 1x8 blocks, each of
  //     them 8x16 pixels.
  //
  //   * a region of 8x160 pixels is a region of 1x20 MCU.  The first
  //     12 MCUs will be merged into four blocks, each the merge of 3
  //     MCU, while the remaining 8 MCUs will be merged in pairs to
  //     form other four blocks.  The output will be 1x8 blocks, the
  //     first four blocks being 8x24 pixels and the other frou being
  //     8x16 pixels.

  // Number of MCU cols to merge for each block
  int mcu_cols_per_block = ncols / 8;
  // Number of MCU cols to merge for the "larger" blocks (if any)
  int mcu_cols_per_xl_block = mcu_cols_per_block +1;
  // Number of blocks that will be merged with an extra MCU
  int n_xl_col_blocks = ncols % 8;
  // Number of MCU cols that are merged into the "larger" blocks
  int mcu_cols_merged_xl = mcu_cols_per_xl_block * n_xl_col_blocks;

  // Idem for rows
  int mcu_rows_per_block = nrows / 8;
  int mcu_rows_per_xl_block = mcu_rows_per_block +1;
  int n_xl_row_blocks = nrows % 8;
  int mcu_rows_merged_xl = mcu_rows_per_xl_block * n_xl_row_blocks;

  // Memory of at most row coeffs to copy across.
  std::vector<JCOEF> row_coeffs (8);

  for (int comp_i = 0; comp_i < src.info.num_components; ++comp_i) {
    // jpeg_component_info *comp_info = srcinfo->comp_info + comp_i;

    // Should be possible to specify the number of rows in
    // access_virt_barray but I keep getting "Bogus virtual array
    // access".  Use a for loop then.
    for (int ri = start_row; ri < start_row + nrows; ++ri) {
      const int rel_ri = ri - start_row;
      bool is_first_row_mcu_in_block;
      if (rel_ri < mcu_rows_merged_xl)
        is_first_row_mcu_in_block = (rel_ri % mcu_rows_per_xl_block) == 0;
      else
        is_first_row_mcu_in_block = ((rel_ri - mcu_rows_merged_xl)
                                     % mcu_rows_per_block) == 0;

      JBLOCKARRAY buf = src.info.mem->access_virt_barray((j_common_ptr)&src.info,
                                                         src_coeffs_array[comp_i],
                                                         ri, 1, TRUE);
      for (int ci = 0; ci < ncols; ++ci) {
        int merged_block_col;
        bool is_first_col_mcu_in_block;
        if (ci < mcu_cols_merged_xl) {
          merged_block_col = ci / mcu_cols_per_xl_block;
          is_first_col_mcu_in_block = (ci % mcu_cols_per_xl_block) == 0;
        } else {
          merged_block_col = (n_xl_col_blocks + ((ci - mcu_cols_merged_xl)
                                                 / mcu_cols_per_block));
          is_first_col_mcu_in_block = ((ci - mcu_cols_merged_xl)
                                       % mcu_cols_per_block) == 0;
        }

        JCOEFPTR coefptr = buf[0][start_col + ci];
        if (is_first_col_mcu_in_block && is_first_row_mcu_in_block)
          row_coeffs[merged_block_col] = coefptr[0];
        else
          coefptr[0]= row_coeffs[merged_block_col];
        for (int i = 1; i < DCTSIZE2; ++i)
          coefptr[i]= 0;
      }
    }
  }
}


int
jpegblur(std::FILE *srcfile, std::FILE *dstfile,
         const std::vector<BoundingBox>& bounding_boxes)
{
  JPEGDecompressor src {srcfile};
  src.info.buffered_image = TRUE;

  // if (src.maybe_has_jfif_thumbnail()) {
  //   std::cerr << "This image might have a thumbnail.\n";
  //   return 1;
  // } else if (src.is_progressive()) {
  //   std::cerr << "This file has progressive mode.\n";
  //   return 1;
  // }

  JPEGCompressor dst {dstfile};

  const cv::Mat img = src.to_image();
  const cv::Mat blurred_img = blur_image(img, bounding_boxes);
  // FIXME: this should be just a JPEGDecompressor, no need to
  // specialized class, go make it virtual.
  JPEGFileDecompressor blurred = JPEGFileDecompressor::from_image(blurred_img,
                                                                  src);

  // TODO: maybe have src and blurred save the image
  merge(src, img, blurred, blurred_img);

  // XXX: the order of operations is quite sensitive to ensure that we
  // get an output file as similar as possible as the input.  To test
  // this, call this program without any bounding box and the output
  // file should be exactly the same, bit by bit, as the input.  And
  // just because it works for one file, does not mean it will work
  // for others so test it in a whole dataset.
  //
  // Some things that I have already found cause issues:
  //
  //   * save_markers must happen before jpeg_read_header
  dst.copy_critical_parameters_from(src);

  // copy_critical_parameters_from copies many things such as the
  // original quantisation tables but not the Huffman tables and
  // others parameters that may be changed without changing decoded
  // pixel values.  However, we want to keep as much as possible, so
  // we copy those ourselves.
  dst.info.optimize_coding = FALSE;
  for (int i = 0; i < NUM_HUFF_TBLS; i++) {
    dst.info.ac_huff_tbl_ptrs[i] = src.info.ac_huff_tbl_ptrs[i];
    dst.info.dc_huff_tbl_ptrs[i] = src.info.dc_huff_tbl_ptrs[i];
  }

  dst.info.arith_code = src.info.arith_code;

  dst.info.restart_interval = src.info.restart_interval;

  dst.copy_coefficients_from(src);
  dst.copy_markers_from(src);

  return 0;
}


enum class CensorType {
   black,
   pixelisation,
   blurring,
};


struct jpegblur_conf {
  CensorType censor_type;
  std::vector<BoundingBox> bounding_boxes;
};


struct jpegblur_conf
parse_cmdline_args (const int argc, char *argv[])
{
  // Defaults:
  struct jpegblur_conf parsed_conf = {
    CensorType::blurring,
    {},
  };

  struct option long_options[] = {
    {"black", no_argument, nullptr, 'k'},
    {"blur", no_argument, nullptr, 'b'},
    {"pixelise", no_argument, nullptr, 'p'},
    {0, 0, 0, 0}
  };

  int option_index = 0;
  while (1) {
    const int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == -1)
      break;
    const char* option_name = long_options[option_index].name;
    if (option_name == "black")
      parsed_conf.censor_type = CensorType::black;
    else if (option_name == "blur")
      parsed_conf.censor_type = CensorType::blurring;
    else if (option_name == "pixelise")
      parsed_conf.censor_type = CensorType::pixelisation;
    else
      abort();
  }

  // Handle non-option ARGV elements (bounding boxes)
  if (optind < argc) {
    while (optind < argc) {
      BoundingBox bbox = BoundingBox::from_cmdline_arg(argv[optind++]);
      parsed_conf.bounding_boxes.push_back(bbox);
    }
  }
 return parsed_conf;
}


int
main(const int argc, char *argv[])
{
  std::FILE *infile = stdin;
  std::FILE *outfile = stdout;

  struct jpegblur_conf conf = parse_cmdline_args(argc, argv);

  if (conf.censor_type == CensorType::black) {
    std::cerr << "Black bars censoring not yet merged in.\n";
    return 1;
  } else if (conf.censor_type == CensorType::pixelisation) {
    std::cerr << "Pixelisation censoring not yet merged in.\n";
    return 1;
  } else if (conf.censor_type == CensorType::blurring) {
    return jpegblur(infile, outfile, conf.bounding_boxes);
  }
}
