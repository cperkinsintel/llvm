
# LINEAR TILING is not working.  There are no errors about it being unsupported,
# it simply doesn't work right.  
./vss_2d_test.bin 16x16 --linear

# This sempahore test normally passes perfectly.  But now it locks up.
./vss_2d_test.bin 4x4 --semaphores