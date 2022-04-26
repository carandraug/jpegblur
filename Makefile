.PHONY: all

all: jpegblur print-jpeg-fields

jpegblur: jpegblur.cc
	c++ -ljpeg -lopencv_core -lopencv_imgproc -lopencv_highgui $< -o $@

print-jpeg-fields: print-jpeg-fields.cc
	c++ -ljpeg $< -o $@
