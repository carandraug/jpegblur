jpegblur: jpegblur.cc
	c++ -ljpeg -lopencv_core -lopencv_imgproc $< -o $@
