#!/bin/sh

# Simple bash shell script to generate Python bindings for OPI
#
# SYNOPSIS:
# Running "sh generate_python_bindings.sh <full path to OPI installation directory>"
#  will create a file called "opi.py" in the current directory
#
# PREREQUISITES:
# -OPI built and installed
# -h2xml and xml2py installed
#
# The Python bindings are generated from the C header files that are created
# during the OPI build process. Therefore, OPI must be built and installed before
# the Python bindings can be created.
# This script uses the tools h2xml and xml2py to generate Python code. On Debian
# and derivates like Ubuntu, they can be found inside the package "python-ctypeslib".
# Use "sudo apt-get install python-ctypeslib" to install them.
# The script needs the full path to the OPI installation directory as a command line
# argument. On successful completion, the file opi.py will appear in the current
# working directory. It can be imported into a Python script and used just like the
# native C API. See the included Python host example for details.
#
# This script will probably be integrated into the build process some time in the future.

if [ -z "$1" ]
then
	echo "Usage: $0 <full path to OPI installation directory>"
elif [ -z `which h2xml` ] || [ -z `which xml2py` ]
then
	echo "Please install h2xml and xml2py (Debian/Ubuntu: apt-get install python-ctypeslib"
else
	OPI_INSTALL=$1
	OUT_FILE=opi.py
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPI_INSTALL/lib
	h2xml $OPI_INSTALL/include/OPI/opi_c_bindings.h -I $OPI_INSTALL/include/OPI/ -q -c -o /tmp/opi.py.xml
	xml2py /tmp/opi.py.xml -o $OUT_FILE -l libOPI.so
	rm /tmp/opi.py.xml
fi
