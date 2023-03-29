bin_PROGRAMS = jpegblur print-jpeg-fields

# XXX: We do the following map:
#
#     program_LDADD = pkgconfig_module_LIBS (pkg-config --libs)
#     program_CXXFLAGS = pkgconfig_module_CFLAGS (pkg-config --cflags)
#
# I think that's not quite right.  From automake documentation:
#
#   > prog_LDADD is inappropriate for passing program-specific linker
#   > flags (except for -l, -L, -dlopen and -dlpreopen). So, use the
#   > prog_LDFLAGS variable for this purpose.
#
# and _CPPFLAGS are the preprocessor flags while CXXFLAGS are the c++
# compiler flags so they should be separated.  With that in mind, I
# think the right mapping is:
#
#     program_LDADD = pkg-config --libs
#     program_LDFLAGS = pkg-config --libs-only-other
#     program_CPPFLAGS = pkg-config --cflags-only-I
#     program_CXXFLAGS = pkg-config --cflags-only-other
#
# However, the macros in pkg.m4 only create the _LIBS and _CFLAGS
# variables so until this is a problem for someone we are not fixing
# this.

jpegblur_SOURCES = jpegblur.cc
jpegblur_LDADD = $(LIBJPEG_LIBS) $(OPENCV_LIBS)
jpegblur_CXXFLAGS = $(LIBJPEG_CFLAGS) $(OPENCV_CFLAGS)

print_jpeg_fields_SOURCES = print-jpeg-fields.cc
print_jpeg_fields_LDADD = $(LIBJPEG_LIBS)
print_jpeg_fields_CXXFLAGS = $(LIBJPEG_CFLAGS)
