## Copyright (C) 2023 David Miguel Susano Pinto <pinto@robots.ox.ac.uk>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

AC_INIT([jpegblur], [1.1+dev])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_MACRO_DIRS([m4])

dnl Make --enable-silent-rules the default.  To get verbose build
dnl output, users can run configure with --disable-silent-rules or use
dnl "make V=1".
AM_SILENT_RULES([yes])


AC_PROG_CXX
AC_LANG(C++)


dnl We don't use CC but it is needed by the pkg-config macros.
AC_PROG_CC

PKG_PROG_PKG_CONFIG


PKG_CHECK_MODULES([LIBJPEG], [libjpeg])

dnl We include cstdio because libjpeg requires us to do it.  As
dnl described in "libjpeg.txt", section "Mechanics of usage: include
dnl files, linking, etc":
dnl
dnl     Before including jpeglib.h, include system headers that define
dnl     at least the typedefs FILE and size_t.  On ANSI-conforming
dnl     systems, including <stdio.h> is sufficient; [...]

AC_CHECK_HEADER([jpeglib.h],
  [],  dnl do nothing different
  [AC_MSG_ERROR([Unable to include jpeglib.h])],
  [[#include <cstdio>]]
)

dnl Since libjpeg v7, jpeg_component_info has separate
dnl DCT_h_scaled_size and DCT_v_scaled_size members instead of a
dnl single DCT_scaled_size member.  Ensure compatibility with libjpeg
dnl v6b (aka v62) which is still common and the default compatibility
dnl for libjpeg-turbo.
AC_CHECK_MEMBER([jpeg_component_info.DCT_scaled_size],
  [AC_DEFINE([HAVE_JPEG_COMPONENT_INFO_DCT_SCALED_SIZE], 1,
             [Define to 1 if `DCT_scaled_size' is a member of `jpeg_component_info'.])],
  [],  dnl HAVE_* will be undef because DCT_scaled has separate hv members
  [[
#include <cstdio>
#include <jpeglib.h>
]]
)

dnl There is a series of issues with getting the configuration needed
dnl for OpenCV:
dnl
dnl   1) Before OpenCV 4, the pkg-config module name was "opencv".
dnl      Since OpenCV 4, the module name is "opencv4".  I guess it's
dnl      expected that future opencv versions will also bump the
dnl      module name so this list of alternative names will need to be
dnl      kept up to date as they make new releases.  Fun!
dnl
dnl   2) There are systems (our Triton cluster for example) where the
dnl      include path provided only includes the deprecated opencv 1
dnl      headers without the new opencv2 directory.  So we really need
dnl      to check if we can find the header file.
dnl
dnl   3) We only need the opencv_core and opencv_imgproc libs but
dnl      there doesn't seem to be a way to get the flags for only
dnl      those modules with the .pc provided.  So we end up linking to
dnl      a lot more than we actually need.

PKG_CHECK_MODULES([OPENCV], [opencv],
  [],  # do nothing different if found
  [PKG_CHECK_MODULES([OPENCV], [opencv4])],
)

dnl AC_CHECK_HEADER 4th arg is a list of extra include directories but
dnl we have a list of CXXFLAGS not a list of directories.  So modify,
dnl temporarily, CXXFLAGS for the test.
save_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS $OPENCV_CFLAGS"
AC_CHECK_HEADER([opencv2/imgproc.hpp],
  [],  dnl for now, do nothing different if found
  [AC_MSG_ERROR([opencv2/imgproc.hpp not found])]
)
CXXFLAGS="$save_CXXFLAGS"


dnl We require GNU's getopt_long
AC_CHECK_HEADER([getopt.h],
  [AC_CHECK_FUNC([getopt_long],
    [],
    [AC_MSG_ERROR([getopt.h was found but getopt_long is not available])],
  )],
  [AC_MSG_ERROR([getopt.h file not found])],
)


AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

AC_OUTPUT
