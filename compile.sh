cmake . -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)")
make -j 48
