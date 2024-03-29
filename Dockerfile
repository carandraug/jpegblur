## Copyright (C) 2022 David Miguel Susano Pinto <carandraug@gmail.com>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

FROM debian:buster

RUN apt-get update \
    && apt-get install -y \
        g++ \
        make \
        libjpeg62-turbo-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/jpegblur

COPY ./ ./

RUN autogen.sh \
    && configure \
    && make \
    && make install
