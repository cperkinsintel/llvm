# Unlike 2D, linear tiling seems to work fine for 1D images.


echo "--- SAMPLED 1D.  OPTIMAL TILING ---"
./vss_1d_test.bin 1025x
./vss_1d_test.bin 1025x --linear

echo "--- SAMPLED 2D.  OPTIMAL TILING w SEMAPHORES ---"
./vss_1d_test.bin 1025x --semaphores
./vss_1d_test.bin 1025x --semaphores --linear

## SAMPLING is for READING. There is no sampled "write". 

echo "--- UNSAMPLED 2D.  OPTIMAL TILING ---"
./vsu_1d_test.bin 1025x
./vsu_1d_test.bin 1025x --linear

echo "--- UNSAMPLED 2D.  OPTIMAL TILING w SEMAPHORES ---"
./vsu_1d_test.bin 1025x --semaphores
./vsu_1d_test.bin 1025x --semaphores --linear


echo "--- UNSAMPLED 2D WRITE.  OPTIMAL TILING ---"
./vsu_1d_w_test.bin 1025x
./vsu_1d_w_test.bin 1025x --linear

echo "--- UNSAMPLED 2D WRITE.  OPTIMAL TILING w SEMAPHORES ---"
./vsu_1d_w_test.bin 1025x --semaphores
./vsu_1d_w_test.bin 1025x --semaphores --linear




