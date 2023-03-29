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

AC_INIT([jpegblur], [1.0])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_MACRO_DIRS([m4])

dnl Make --enable-silent-rules the default.  To get verbose build
dnl output, users can run configure with --disable-silent-rules or use
dnl "make V=1".
AM_SILENT_RULES([yes])

AC_PROG_CXX

dnl While we don't use CC directly, it is needed by the pkg-config
dnl macros.
AC_PROG_CC

PKG_PROG_PKG_CONFIG

PKG_CHECK_MODULES([LIBJPEG], [libjpeg])

dnl We only need opencv_core and opencv_imgproc but upstream does
dnl there doesn't seem to be a way to get the flags for only those
dnl modules with the .pc provided.  This will link to a lot more than
dnl we actually need.
PKG_CHECK_MODULES([OPENCV], [opencv])


AC_CONFIG_FILES([
 Makefile
 src/Makefile
])

AC_OUTPUT
