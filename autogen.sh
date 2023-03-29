#!/bin/sh

# This script is to run when jpegblur checked out from git sources.
# It's kinda pointless, but I can never remember that it's autoreconf
# that I need to run and autogen.sh was how it was done in the old
# times.

set -o errexit  # Don't ignore failures.

autoreconf --install
echo "autogen.sh done.  Now you can run './configure'."
