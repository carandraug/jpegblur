## Copyright (C) 2022 David Miguel Susano Pinto <carandraug@gmail.com>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

FROM debian:buster

RUN apt-get update \
    && apt-get install -y \
        autoconf \
        g++ \
        make \
        libjpeg62-turbo-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/jpegblur

COPY ./ ./

## Only run autogen.sh if we're building from development sources.  If
## we're building from a distribution tarball, autogen.sh is not even
## there.
RUN if [ -e "autogen.sh" ]; then \
      ./autogen.sh; \
    fi \
    && ./configure \
    && make \
    && make install
