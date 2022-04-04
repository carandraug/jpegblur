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

// NAME
//
//     jpegblur - "lossless" "blurring" of regions in JPEG image
//
// SYNOPSIS
//
//     jpegblur [BOUNDING-BOX ...] < IN_JPEG > OUT_JPEG
//
// DESCRIPTION
//
//     jpegblur takes bounding boxes from command line and returns a
//     JPEG image with those regions blurred.  It does this in DCT
//     space thus avoiding the introduction of new compression
//     artefacts.  It also preserves all metadata by copying all extra
//     markers.
//
//     Regions to be blurred are expanded to include all MCUs (Minimum
//     Coded Unit, the 8x8 pixel squares) under them.  Blurring is
//     done by keeping the DC coefficient of each component and
//     zeroing all AC coefficients, effectively generating a block
//     with the base hue.  In addition, multiple MCUs are merged so
//     that each region will display at most 8 blocks across each
//     axis.
//
//     Regions are defined with `xi,yi,width,height` in pixels units.
//     Multiple regions can be defined.
//
//     Metadata is copied intact.  However, if a JFIF thumbnail is
//     identified jpegblu will error since blurring of the same region
//     in the thumbnail is not implemented.  For the same reason,
//     jpegblur will fail if the file uses progressive mode.
//
//     Ideally, calling `jpegblur` without any region should output a
//     file that is exactly the same byte by byte.  However, encoding
//     settings are not readable and may be tricky to reproduce.
//     Recommend to test this first and adjust source as required.
//
// EXAMPLES
//
//     Blur the top left 10x10 pixels (will be expanded to blur the
//     top-left 16x16 pixels):
//
//         jpegblur 0,0,10,10 < foo.jpg > bar.jpg
//
//     Blur the top left 10x10 pixels (will be expanded to blur the
//     top-left 16x16 pixels) and a 200 pixels wide row in the middle
//     of the image (will be expanded to a 8x200 pixels):
//
//         jpegblur 0,0,10,10 500,500,200,1 < foo.jpg > bar.jpg
//
//    Copy the image without any blurring (ideally it should produce
//    the same file as the input):
//
//         jpegblur < foo.jpg > bar.jpg
//         md5sum foo.jpg bar.jpg

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <jpeglib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>


class BoundingBox {
public:
  const int xi;
  const int yi;
  const int width;
  const int height;

  BoundingBox(const int xi, const int yi, const int width, const int height)
     : xi{xi}, yi{yi}, width{width}, height{height}
  {}

  static BoundingBox
  from_cmdline_arg(const std::string& arg)
  {
    int xi, yi, width, height;
    std::stringstream argss(arg);
    std::string sep("   ");
    if (! (argss >> xi >> sep[0]
           >> yi >> sep[1]
           >> width >> sep[2]
           >> height)
        || sep != ",,,"
        || ! argss.eof())
      throw std::invalid_argument("failed to parse BB from '" + arg + "'");
  return BoundingBox{xi, yi, width, height};
  }

  // New BB expanded to include all MCUs under the original BB.
  BoundingBox
  expanded_to_MCU() const
  {
    const int x_xi = xi /8 *8;
    const int x_yi = yi /8 *8;
    const int x_width = (width + xi - x_xi + (8 -1)) /8 *8;
    const int x_height = (height + yi - x_yi + (8 -1)) /8 *8;
    return BoundingBox{x_xi, x_yi, x_width, x_height};
  }
};


class JPEGDecompressor {
private:
  struct jpeg_error_mgr jerr;

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
    info.buffered_image = TRUE;

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
    // Should be possible to specify the number of rows in
    // access_virt_barray but I keep getting "Bogus virtual array
    // access".
    return info.mem->access_virt_barray((j_common_ptr)&info,
                                        coeff_arrays[component_idx],
                                        row_idx, 1, TRUE);
  }

  cv::Mat
  to_image()
  {
    const std::vector<int> sz{info.image_height,
                              info.image_width,
                              info.num_components};

    cv::Mat rgb;
    if (sizeof(JSAMPLE) == 1)
      rgb = cv::Mat(sz, CV_8UC1);
    else  // libjpeg built with BITS_IN_JSAMPLE == 12
      rgb = cv::Mat(sz, CV_16UC1);

    // We are not dealing with progressive images and only use
    // buffered-image to keep all coefficients in memory.  So we can
    // just specify scan=1 and assume we are dealing with the full
    // resolution image/scan.
    jpeg_start_decompress(&info);
    jpeg_start_output(&info, 1);

    const int row_stride = rgb.size[1] * rgb.size[2];
    JSAMPARRAY buffer = info.mem->alloc_sarray((j_common_ptr)&info,
                                               JPOOL_IMAGE, row_stride, 1);

    int line_i = 0;
    while (info.output_scanline < info.output_height) {
      jpeg_read_scanlines(&info, buffer, 1);
      for (int i = 0; i < row_stride; i++)
        rgb.data[line_i*row_stride + i] = buffer[0][i];

      ++line_i;
    }

    jpeg_finish_output(&info);
    return rgb;
  }

  ~JPEGDecompressor()
  {
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
  const std::string fpath;

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
      dst.info.in_color_space = JCS_RGB;
      dst.info.optimize_coding = TRUE;
      jpeg_start_compress(&dst.info, TRUE);

      JSAMPROW row_pointer[1];
      const int row_stride = img.size[1] * img.size[2];
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


// Blur image with a kernel appropriate for the largest bounding box.
cv::Mat
blur_image(const cv::Mat& src, const std::vector<BoundingBox>& bounding_boxes)
{
  std::vector<int> lengths;
  std::vector<double> sigmas;
  int length = 0;
  double sigma = 0.0;
  for (auto bb : bounding_boxes) {
    length = std::max<int>({length, bb.width /2, bb.height /2});
    sigma = std::max<double>({sigma, bb.width /10., bb.height /10.});
  }
  if (length % 2 == 0)  // OpenCV requires kernel length to be odd
    ++length;

  // Gaussian blur only handles 2D arrays so merge the channels.
  cv::Mat dst = src.clone();
  cv::GaussianBlur(src.reshape(3, 2, src.size.p),
                   dst.reshape(3, 2, src.size.p),
                   cv::Size(length, length), sigma, sigma,
                   cv::BorderTypes::BORDER_REPLICATE);
  return dst;
}


void
copy_region_from(JPEGDecompressor& dst, JPEGFileDecompressor& blurred,
                 const BoundingBox& bb)
{
  // We are doing the indexing in MCU coordinates and not in pixels
  // (one MCU corresponds to 8x8 pixels).
  const int start_col = bb.xi / 8;
  const int end_col = start_col + (bb.width /8);

  const int start_row = bb.yi / 8;
  const int end_row = start_row + (bb.height /8);

  for (int comp_i = 0; comp_i < dst.info.num_components; ++comp_i) {
    for (int row_i = start_row; row_i < end_row+1; ++row_i) {
      JBLOCKARRAY dst_buf = dst.get_coeff_for_row(comp_i, row_i);
      JBLOCKARRAY src_buf = blurred.get_coeff_for_row(comp_i, row_i);

      for (int col_i = start_col; col_i < end_col +1; ++col_i)
        for (int i = 0; i < DCTSIZE2; ++i)
          dst_buf[0][col_i][i] = src_buf[0][col_i][i];
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
  const int start_col = mcu_bb.xi / 8;
  const int end_col = start_col + (mcu_bb.width /8);
  const int ncols = end_col - start_col;

  const int start_row = mcu_bb.yi / 8;
  const int end_row = start_row + (mcu_bb.height /8);
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
  // The rest of the code expects bounding boxes to lie on MCU limits.
  // We expand it here but they don't do any check.  We should have a
  // separate type for such MCU expanded BB.
  std::vector<BoundingBox> xl_bounding_boxes;
  for (auto bb : bounding_boxes)
    xl_bounding_boxes.push_back(bb.expanded_to_MCU());

  JPEGDecompressor src {srcfile};

  if (src.maybe_has_jfif_thumbnail()) {
    std::cerr << "This image might have a thumbnail.\n";
    return 1;
  } else if (src.is_progressive()) {
    std::cerr << "This file has progressive mode.\n";
    return 1;
  }

  JPEGCompressor dst {dstfile};

  if (xl_bounding_boxes.size()) {
    const cv::Mat blurred_img = blur_image(src.to_image(), xl_bounding_boxes);
    // FIXME: this should be just a JPEGDecompressor, no need to
    // specialized class, go make it virtual.
    JPEGFileDecompressor blurred = JPEGFileDecompressor::from_image(blurred_img,
                                                                    src);

    for (auto bb : xl_bounding_boxes)
      copy_region_from(src, blurred, bb);
  }

  // XXX: the order of operations is quite sensitive to ensure that we
  // get an output file as similar as possible as the input.  To test
  // this, call this program without any bounding box and the output
  // file should be exactly the same, bit by bit, as the input.  And
  // just because it works for one file, does not mean it will work
  // for others so test it in a whole dataset.
  //
  // Some things that I have already found cause issues:
  //
  //   * Setting dstinfo.optimize_coding must happen after calling
  //     jpeg_copy_critical_parameters
  //
  //   * save_markers must happen before jpeg_read_header
  dst.copy_critical_parameters_from(src);

  // We don't know if coding was optimised on the input file.  Ideally
  // we would use exactly the same options but failing that, let us
  // optimise for disk space.
  dst.info.optimize_coding = TRUE;

  dst.info.arith_code = src.info.arith_code;

  dst.copy_coefficients_from(src);
  dst.copy_markers_from(src);

  return 0;
}


int
main(int argc, char *argv[])
{
  std::FILE *infile = stdin;
  std::FILE *outfile = stdout;

  std::vector<BoundingBox> bounding_boxes;
  bool do_pixelation = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg {argv[i]};
    if (i == 1 && arg == "--pixelate")
      do_pixelation = true;
    else
      bounding_boxes.push_back(BoundingBox::from_cmdline_arg(arg));
  }

  if (do_pixelation) {
    std::cerr << "Pixelation code path not merged in yet.\n";
    return 1;
  }

  return jpegblur(infile, outfile, bounding_boxes);
}
