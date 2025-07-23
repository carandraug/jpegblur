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

// This program prints all the values in a jpeg_decompress_struct and
// associated jpeg_component_info.  It is used for debugging.

#include <cstdio>  // required by jpeglib.h
#include <iostream>
#include <jpeglib.h>

std::string
colorspace_to_string(int color_space)
{
#define CASE(NAME) \
  case NAME: return #NAME

  switch (color_space) {
    CASE(JCS_UNKNOWN);
    CASE(JCS_GRAYSCALE);
    CASE(JCS_RGB);
    CASE(JCS_YCbCr);
    CASE(JCS_CMYK);
    CASE(JCS_YCCK);
    CASE(JCS_EXT_RGB);
    CASE(JCS_EXT_RGBX);
    CASE(JCS_EXT_BGR);
    CASE(JCS_EXT_BGRX);
    CASE(JCS_EXT_XBGR);
    CASE(JCS_EXT_XRGB);
    CASE(JCS_EXT_RGBA);
    CASE(JCS_EXT_BGRA);
    CASE(JCS_EXT_ABGR);
    CASE(JCS_EXT_ARGB);
    CASE(JCS_RGB565);
  default:
    return "error: unlisted";
  };
}


std::string
dct_method_to_string(int dct_method)
{
#define CASE(NAME) \
  case NAME: return #NAME

  switch (dct_method) {
    CASE(JDCT_ISLOW);
    CASE(JDCT_IFAST);
    CASE(JDCT_FLOAT);
  default:
    return "error: unlisted";
  };
}


std::string
dither_method_to_string(int dither_method)
{
#define CASE(NAME) \
  case NAME: return #NAME

  switch (dither_method) {
    CASE(JDITHER_NONE);
    CASE(JDITHER_ORDERED);
    CASE(JDITHER_FS);
  default:
    return "error: unlisted";
  };
}


int
main(int argc, char *argv[])
{
  std::FILE *infile = stdin;
  struct jpeg_error_mgr jerr;
  struct jpeg_decompress_struct info;

  info.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&info);
  jpeg_stdio_src(&info, infile);

  jpeg_read_header(&info, TRUE);
  jpeg_start_decompress(&info);

#define PRINT_FIELD_VALUE(NAME, VAL) \
  std::cout << "info." << #NAME <<": " << VAL << "\n"

#define PF(NAME) \
  PRINT_FIELD_VALUE(NAME, info.NAME)

#define PF_UINT8(NAME) \
  PRINT_FIELD_VALUE(NAME, static_cast<int>(info.NAME))

#define PF_CS(NAME) \
  PRINT_FIELD_VALUE(NAME, colorspace_to_string(info.NAME))

#define PF_bool(NAME) \
  PRINT_FIELD_VALUE(NAME, (info.NAME ? "True" : "False"))

#define PF_DCT(NAME) \
  PRINT_FIELD_VALUE(NAME, dct_method_to_string(info.NAME))

#define PF_DITHER(NAME) \
  PRINT_FIELD_VALUE(NAME, dither_method_to_string(info.NAME))

  PF(image_width);
  PF(image_height);
  PF(num_components);
  PF_CS(jpeg_color_space);
  PF_CS(out_color_space);
  PF(scale_num);
  PF(scale_denom);
  PF(output_gamma);
  PF_bool(buffered_image);
  PF_bool(raw_data_out);
  PF_DCT(dct_method);

  PF_bool(do_fancy_upsampling);
  PF_bool(do_block_smoothing);

  PF_bool(quantize_colors);
  // the following are ignored if not quantize_colors:
  PF_DITHER(dither_mode);
  PF_bool(two_pass_quantize);
  PF(desired_number_of_colors);

  // these are significant only in buffered-image mode:
  PF_bool(enable_1pass_quant);
  PF_bool(enable_external_quant);
  PF_bool(enable_2pass_quant);

  PF(output_width);
  PF(output_height);
  PF(out_color_components);
  PF(output_components);
  PF(rec_outbuf_height);

  PF(actual_number_of_colors);

  PF_bool(progressive_mode);
  PF_bool(arith_code);
  PF(restart_interval);
  PF_bool(saw_JFIF_marker);
  PF_UINT8(JFIF_major_version);
  PF_UINT8(JFIF_minor_version);
  PF_UINT8(density_unit);
  PF(X_density);
  PF(Y_density);
  PF_bool(saw_Adobe_marker);
  PF_UINT8(Adobe_transform);
  PF_bool(CCIR601_sampling);
  PF(total_iMCU_rows);
  PF(MCUs_per_row);
  PF(MCU_rows_in_scan);
  PF(blocks_in_MCU);

  for (int i = 0; i < info.num_components; i++) {

#define PRINT_COMP_FIELD_VALUE(NAME) \
    std::cout << "info.comp_info[" << i << "]." << #NAME <<": " \
              << info.comp_info[i].NAME << "\n"

    PRINT_COMP_FIELD_VALUE(component_id);
    PRINT_COMP_FIELD_VALUE(component_index);
    PRINT_COMP_FIELD_VALUE(h_samp_factor);
    PRINT_COMP_FIELD_VALUE(v_samp_factor);
    PRINT_COMP_FIELD_VALUE(quant_tbl_no);
    PRINT_COMP_FIELD_VALUE(dc_tbl_no);
    PRINT_COMP_FIELD_VALUE(ac_tbl_no);
    PRINT_COMP_FIELD_VALUE(width_in_blocks);
    PRINT_COMP_FIELD_VALUE(height_in_blocks);
    PRINT_COMP_FIELD_VALUE(DCT_scaled_size);
    PRINT_COMP_FIELD_VALUE(downsampled_width);
    PRINT_COMP_FIELD_VALUE(downsampled_height);
    PRINT_COMP_FIELD_VALUE(component_needed);
    PRINT_COMP_FIELD_VALUE(MCU_width);
    PRINT_COMP_FIELD_VALUE(MCU_height);
    PRINT_COMP_FIELD_VALUE(MCU_blocks);
    PRINT_COMP_FIELD_VALUE(MCU_sample_width);
    PRINT_COMP_FIELD_VALUE(last_col_width);
    PRINT_COMP_FIELD_VALUE(last_row_height);
  }

  jpeg_destroy_decompress(&info);
  return 0;
}
