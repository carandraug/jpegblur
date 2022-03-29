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

#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <jpeglib.h>


typedef struct {
  int xi;
  int yi;
  int width;
  int height;
} BoundingBox;


BoundingBox
parse_bounding_box(const std::string arg)
{
  BoundingBox bb;
  std::stringstream argss (arg);
  std::string sep ("   ");
  if (! (argss >> bb.xi >> sep[0]
               >> bb.yi >> sep[1]
               >> bb.width >> sep[2]
               >> bb.height)
      || sep != ",,,"
      || ! argss.eof())
    throw std::invalid_argument("failed to parse BB from '" + arg + "'");
  return bb;
}



// Setup decompression object to save desired markers in memory.
void
save_markers(jpeg_decompress_struct *srcinfo)
{
  jpeg_save_markers(srcinfo, JPEG_COM, 0xFFFF);
  for (int m = 0; m < 16; m++)
    jpeg_save_markers(srcinfo, JPEG_APP0 + m, 0xFFFF);
}


// Copy markers saved in the given source object to the destination object.
//
// This should be called just after jpeg_start_compress() or
// jpeg_write_coefficients().  Note that those routines will have
// written the SOI, and also the JFIF APP0 or Adobe APP14 markers if
// selected so we need to skip those.
void
copy_markers(jpeg_decompress_struct *srcinfo, jpeg_compress_struct *dstinfo)
{
  jpeg_saved_marker_ptr marker;
  for (marker = srcinfo->marker_list; marker != NULL; marker = marker->next) {
    if (dstinfo->write_JFIF_header &&
        marker->marker == JPEG_APP0 &&
        marker->data_length >= 5 &&
        marker->data[0] == 0x4A &&
        marker->data[1] == 0x46 &&
        marker->data[2] == 0x49 &&
        marker->data[3] == 0x46 &&
        marker->data[4] == 0)
      continue;                 // reject duplicate JFIF
    if (dstinfo->write_Adobe_marker &&
        marker->marker == JPEG_APP0 + 14 &&
        marker->data_length >= 5 &&
        marker->data[0] == 0x41 &&
        marker->data[1] == 0x64 &&
        marker->data[2] == 0x6F &&
        marker->data[3] == 0x62 &&
        marker->data[4] == 0x65)
      continue;                 // reject duplicate Adobe
    jpeg_write_marker(dstinfo, marker->marker, marker->data,
                      marker->data_length);
  }
}


void
blur_regions(jpeg_decompress_struct *srcinfo,
             jvirt_barray_ptr *src_coeffs_array,
             const BoundingBox& bb)
{
  // We are doing the indexing in MCU coordinates and not in pixels
  // (one MCU corresponds to 8x8 pixels).
  int start_col = bb.xi / 8;
  int ncols = bb.width / 8;
  if (bb.width % 8 != 0)
    ncols++;

  int start_row = bb.yi / 8;
  int nrows = bb.height / 8;
  if (bb.height % 8 != 0)
    nrows++;

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

  for (int comp_i = 0; comp_i < srcinfo->num_components; ++comp_i) {
    // jpeg_component_info *comp_info = srcinfo->comp_info + comp_i;

    // Should be possible to specify the number of rows in
    // access_virt_barray but I keep getting "Bogus virtual array
    // access".  Use a for loop then.
    for (int ri = start_row; ri < start_row + nrows; ++ri) {
      int rel_ri = ri - start_row;
      bool is_first_row_mcu_in_block;
      if (rel_ri < mcu_rows_merged_xl)
        is_first_row_mcu_in_block = (rel_ri % mcu_rows_per_xl_block) == 0;
      else
        is_first_row_mcu_in_block = ((rel_ri - mcu_rows_merged_xl)
                                     % mcu_rows_per_block) == 0;

      JBLOCKARRAY buf = srcinfo->mem->access_virt_barray((j_common_ptr)srcinfo,
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
  // Setup source
  struct jpeg_decompress_struct srcinfo;
  struct jpeg_error_mgr srcjerr;
  srcinfo.err = jpeg_std_error(&srcjerr);
  jpeg_create_decompress(&srcinfo);
  jpeg_stdio_src(&srcinfo, srcfile);

  // Setup destination
  struct jpeg_compress_struct dstinfo;
  struct jpeg_error_mgr dstjerr;
  dstinfo.err = jpeg_std_error(&dstjerr);
  jpeg_create_compress(&dstinfo);
  jpeg_stdio_dest(&dstinfo, dstfile);

  // XXX: the order of operations is quite sensitive to ensure that we
  // get an output file as similar as possible as the input.  To test
  // this, call this program without any bounding box and the output
  // file should be exactly the same, bit by bit, as the input.  And
  // just because it works for one file, does not mean it will work
  // for others so test it in a whole dataset.
  //
  // To be honest, why this order is required is not clear to me and
  // this copied is mainly lifted from libjpeg's `jpegtran.c`.  The
  // order is:
  //
  //   1. save_markers
  //   2. jpeg_read_header
  //   3. jpeg_copy_critical_parameters
  //   4. set dstinfo.optimize_coding
  //   5. jpeg_wrie_coefficients
  //   6. copy_markers
  //
  // Some things that I have already found cause issues:
  //
  //   * Setting dstinfo.optimize_coding must happen after calling
  //     jpeg_copy_critical_parameters
  //
  //   * save_markers must happen before jpeg_read_header

  save_markers(&srcinfo);

  jpeg_read_header(&srcinfo, TRUE);
  if (srcinfo.JFIF_minor_version > 1) {
    // Being a JFIF version 1.02, this file might have a thumbnail.
    // We don't support blurring the thumbnail as well.
    fputs("This being JFIF>=1.02 might have a thumbnail.\n", stderr);
    // TODO: clear up memory
    return 1;
  } else if (srcinfo.progressive_mode) {
    fputs("This file has progressive mode.\n", stderr);
    // TODO: clear up memory
    return 1;
  }

  jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(&srcinfo);
  jpeg_copy_critical_parameters(&srcinfo, &dstinfo);

  for (auto bb : bounding_boxes)
    blur_regions(&srcinfo, src_coef_arrays, bb);

  // We don't know if coding was optimised on the input file.  Ideally
  // we would use exactly the same options but failing that, let us
  // optimise for disk space.
  dstinfo.optimize_coding = TRUE;

  dstinfo.arith_code = srcinfo.arith_code;

  jpeg_write_coefficients(&dstinfo, src_coef_arrays);

  copy_markers(&srcinfo, &dstinfo);

  jpeg_finish_compress(&dstinfo);
  jpeg_destroy_compress(&dstinfo);

  jpeg_finish_decompress(&srcinfo);
  jpeg_destroy_decompress(&srcinfo);

  return 0;
}


int
main(int argc, char *argv[])
{
  std::FILE *infile = stdin;
  std::FILE *outfile = stdout;

  std::vector<BoundingBox> bounding_boxes;
  for (int i = 1; i < argc; ++i)
    bounding_boxes.push_back(parse_bounding_box(argv[i]));

  return jpegblur(infile, outfile, bounding_boxes);
}
