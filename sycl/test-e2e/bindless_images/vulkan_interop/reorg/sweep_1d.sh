# Unlike 2D, linear tiling seems to work fine for 1D images.


echo "--- SAMPLED 1D. 4 channels. OPTIMAL & LINEAR TILING ---"
./vss_1d_test.bin 1025x
./vss_1d_test.bin 1025x --linear

echo "--- SAMPLED 1D. 4 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vss_1d_test.bin 1025x --semaphores
./vss_1d_test.bin 1025x --semaphores --linear

## SAMPLING is for READING. There is no sampled "write". 

echo "--- UNSAMPLED 1D. 4 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_test.bin 1025x
./vsu_1d_test.bin 1025x --linear

echo "--- UNSAMPLED 1D. 4 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_test.bin 1025x --semaphores
./vsu_1d_test.bin 1025x --semaphores --linear


echo "--- UNSAMPLED 1D WRITE. 4 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_w_test.bin 1025x
./vsu_1d_w_test.bin 1025x --linear

echo "--- UNSAMPLED 1D WRITE. 4 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_w_test.bin 1025x --semaphores
./vsu_1d_w_test.bin 1025x --semaphores --linear




echo "--- SAMPLED 1D. 2 channels. OPTIMAL & LINEAR TILING ---"
./vss_1d_test.bin 1025x --channels 2
./vss_1d_test.bin 1025x --channels 2 --linear

echo "--- SAMPLED 1D. 2 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vss_1d_test.bin 1025x --channels 2 --semaphores
./vss_1d_test.bin 1025x --channels 2 --semaphores --linear

## SAMPLING is for READING. There is no sampled "write". 

echo "--- UNSAMPLED 1D. 2 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_test.bin 1025x --channels 2
./vsu_1d_test.bin 1025x --channels 2 --linear

echo "--- UNSAMPLED 1D. 2 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_test.bin 1025x --channels 2 --semaphores
./vsu_1d_test.bin 1025x --channels 2 --semaphores --linear


echo "--- UNSAMPLED 1D WRITE. 2 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_w_test.bin 1025x --channels 2
./vsu_1d_w_test.bin 1025x --channels 2 --linear

echo "--- UNSAMPLED 1D WRITE. 2 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_w_test.bin 1025x --channels 2 --semaphores
./vsu_1d_w_test.bin 1025x --channels 2 --semaphores --linear




echo "--- SAMPLED 1D. 1 channels. OPTIMAL & LINEAR TILING ---"
./vss_1d_test.bin 1025x --channels 1
./vss_1d_test.bin 1025x --channels 1 --linear

echo "--- SAMPLED 1D. 1 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vss_1d_test.bin 1025x --channels 1 --semaphores
./vss_1d_test.bin 1025x --channels 1 --semaphores --linear

## SAMPLING is for READING. There is no sampled "write". 

echo "--- UNSAMPLED 1D. 1 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_test.bin 1025x --channels 1
./vsu_1d_test.bin 1025x --channels 1 --linear

echo "--- UNSAMPLED 1D. 1 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_test.bin 1025x --channels 1 --semaphores
./vsu_1d_test.bin 1025x --channels 1 --semaphores --linear


echo "--- UNSAMPLED 1D WRITE. 1 channels.  OPTIMAL & LINEAR TILING ---"
./vsu_1d_w_test.bin 1025x --channels 1
./vsu_1d_w_test.bin 1025x --channels 1 --linear

echo "--- UNSAMPLED 1D WRITE. 1 channels.  OPTIMAL & LINEAR TILING w SEMAPHORES ---"
./vsu_1d_w_test.bin 1025x --channels 1 --semaphores
./vsu_1d_w_test.bin 1025x --channels 1 --semaphores --linear



