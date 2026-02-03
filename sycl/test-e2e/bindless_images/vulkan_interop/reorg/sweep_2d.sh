echo "--- SAMPLED 2D.  OPTIMAL TILING ---"
./vss_2d_test.bin 4x4
./vss_2d_test.bin 4x3
./vss_2d_test.bin 8x4
./vss_2d_test.bin 16x16
./vss_2d_test.bin 17x17

echo "--- SAMPLED 2D.  OPTIMAL TILING w SEMAPHORES ---"
./vss_2d_test.bin 4x4 --semaphores
./vss_2d_test.bin 4x3 --semaphores
./vss_2d_test.bin 8x4 --semaphores
./vss_2d_test.bin 16x16 --semaphores
./vss_2d_test.bin 17x17 --semaphores

## SAMPLING is for READING. There is no sampled "write". 

echo "--- UNSAMPLED 2D.  OPTIMAL TILING ---"
./vsu_2d_test.bin 4x4
./vsu_2d_test.bin 4x3
./vsu_2d_test.bin 8x4
./vsu_2d_test.bin 16x16
./vsu_2d_test.bin 17x17

echo "--- UNSAMPLED 2D.  OPTIMAL TILING w SEMAPHORES ---"
./vsu_2d_test.bin 4x4 --semaphores
./vsu_2d_test.bin 4x3 --semaphores
./vsu_2d_test.bin 8x4 --semaphores
./vsu_2d_test.bin 16x16 --semaphores
./vsu_2d_test.bin 17x17 --semaphores


echo "--- UNSAMPLED 2D WRITE.  OPTIMAL TILING ---"
./vsu_2d_w_test.bin 4x4
./vsu_2d_w_test.bin 4x3
./vsu_2d_w_test.bin 8x4
./vsu_2d_w_test.bin 16x16
./vsu_2d_w_test.bin 17x17

echo "--- UNSAMPLED 2D WRITE.  OPTIMAL TILING w SEMAPHORES ---"
./vsu_2d_w_test.bin 4x4 --semaphores
./vsu_2d_w_test.bin 4x3 --semaphores
./vsu_2d_w_test.bin 8x4 --semaphores
./vsu_2d_w_test.bin 16x16 --semaphores
./vsu_2d_w_test.bin 17x17 --semaphores


# echo "--- LINEAR TILING ---"
# linear tiling does not throw any sort of error , but it simply does not work correctly.
# it ALSO breaks things. Semaphores, for example, lock up entirely after a linear call.
# ./vss_2d_test.bin 4x4 --linear
# ./vss_2d_test.bin 4x3 --linear
# ./vss_2d_test.bin 8x4 --linear
# ./vss_2d_test.bin 16x16 --linear   # <-- this run is VERY SLOW.
# ./vss_2d_test.bin 17x17 --linear  # level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)

