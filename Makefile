jpegblur: jpegblur.cc
	c++ -ljpeg -lopencv_core -lopencv_imgproc -lopencv_highgui $< -o $@
