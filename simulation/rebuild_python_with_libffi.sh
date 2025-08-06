#!/bin/bash

set -e  # Exit on any error

# Define paths
INSTALL_DIR="$HOME"
LIBFFI_PREFIX="$INSTALL_DIR/local/libffi"
# OPENSSL_PREFIX="$INSTALL_DIR/local/openssl"
OPENSSL_PREFIX="$INSTALL_DIR/openssl-1.1.1"
PYTHON_PREFIX="$INSTALL_DIR/python/3.12.3"
BUILD_DIR="$INSTALL_DIR/build_fix_ctypes"
PYTHON_TARBALL="Python-3.12.3.tgz"
PYTHON_SRC_DIR="Python-3.12.3"

mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "🔍 Checking libffi installation..."
if [ ! -f "$LIBFFI_PREFIX/include/ffi.h" ]; then
  echo "⚙️ Rebuilding libffi..."
  wget -nc ftp://sourceware.org/pub/libffi/libffi-3.4.4.tar.gz
  tar -xvzf libffi-3.4.4.tar.gz
  cd libffi-3.4.4
  ./configure --prefix=$LIBFFI_PREFIX
  make -j4
  make install
  cd ..
else
  echo "✅ libffi is already installed."
fi

echo "📦 Downloading Python source if needed..."
wget -nc https://www.python.org/ftp/python/3.12.3/$PYTHON_TARBALL
tar -xvzf $PYTHON_TARBALL

echo "🧹 Cleaning previous Python builds..."
cd $PYTHON_SRC_DIR
make distclean || true

echo "⚙️ Configuring Python with OpenSSL and libffi..."
./configure \
  --prefix=$PYTHON_PREFIX \
  --enable-optimizations \
  --with-openssl=$OPENSSL_PREFIX \
  --with-openssl-rpath=auto \
  CPPFLAGS="-I$LIBFFI_PREFIX/include -I$OPENSSL_PREFIX/include" \
  LDFLAGS="-L$LIBFFI_PREFIX/lib -L$OPENSSL_PREFIX/lib" \
  PKG_CONFIG_PATH="$LIBFFI_PREFIX/lib/pkgconfig"

echo "🛠️ Building Python..."
make -j4
make install

echo "✅ Build complete. Verifying _ctypes module..."
$PYTHON_PREFIX/bin/python3 -c "import _ctypes; print('✅ _ctypes module is available!')"

echo "✅ Python 3.12.3 is installed at: $PYTHON_PREFIX"
echo "👉 To use it, run:"
echo "export PATH=\"$PYTHON_PREFIX/bin:\$PATH\""
echo "👉 Or add the above line to your shell config file (e.g., ~/.bashrc or ~/.bash_profile)."