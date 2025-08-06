#!/bin/bash

# Set installation directories
INSTALL_PREFIX=$HOME
LIBFFI_PREFIX=$INSTALL_PREFIX/local/libffi
OPENSSL_PREFIX=$INSTALL_PREFIX/local/openssl
PYTHON_PREFIX=$INSTALL_PREFIX/python/3.12.3

# Create working directory
mkdir -p $INSTALL_PREFIX/build
cd $INSTALL_PREFIX/build

echo "Downloading and installing libffi..."
# Install libffi
wget ftp://sourceware.org/pub/libffi/libffi-3.4.4.tar.gz
tar -xvzf libffi-3.4.4.tar.gz
cd libffi-3.4.4
./configure --prefix=$LIBFFI_PREFIX
make -j4
make install
cd ..

echo "Downloading and installing OpenSSL..."
# Install OpenSSL
wget https://www.openssl.org/source/openssl-3.2.1.tar.gz
tar -xvzf openssl-3.2.1.tar.gz
cd openssl-3.2.1
./Configure --prefix=$OPENSSL_PREFIX --openssldir=$OPENSSL_PREFIX shared
make -j4
make install
cd ..

echo "Downloading and installing Python 3.12.3..."
# Install Python
wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
tar -xvzf Python-3.12.3.tgz
cd Python-3.12.3

./configure --prefix=$PYTHON_PREFIX \
            --enable-optimizations \
            --with-openssl=$OPENSSL_PREFIX \
            --with-openssl-rpath=auto \
            CPPFLAGS="-I$LIBFFI_PREFIX/include -I$OPENSSL_PREFIX/include" \
            LDFLAGS="-L$LIBFFI_PREFIX/lib -L$OPENSSL_PREFIX/lib"

make -j4
make install

echo "✅ Python 3.12.3 has been installed in $PYTHON_PREFIX"

# Optional: Add to PATH for immediate use
echo "To use this Python, add the following to your shell config:"
echo "export PATH=\"$PYTHON_PREFIX/bin:\$PATH\""
