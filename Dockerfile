## Copyright 2022 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

FROM debian:bookworm

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
