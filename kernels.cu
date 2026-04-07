/* This file contains CUDA functions for implementing deep learning network layers.

E6692 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FILE TO COMPLETE THE ASSIGNMENT
*/

#define BLOCK_SIZE 32


__global__ void transpose(float* input, float* output, int outWidth, int outHeight){
    /* 
        Transposes input and writes the result to output.
    
        params:
           input: pointer to the input matrix to be transposed
           output: pointer to the output transposed matrix
           outWidth: the output width in number of columns
           outHeight: the input height in number of rows
    */

    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE+1];
    
    // read the matrix tile into shared memory
    // load one element per thread from device memory (input) and store it
    // in transposed order in block[][]
    
    unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    if(xIndex < outWidth && yIndex < outHeight){
        unsigned int index_in = yIndex * outWidth + xIndex;
        block[threadIdx.y][threadIdx.x] = input[index_in];
    }

    // synchronize to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (output) in linear order
    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if(xIndex < outHeight && yIndex < outWidth){
        unsigned int index_out = yIndex * outHeight + xIndex;
        output[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


__global__ void add(float* A, float* B, int size){
    /*
        Performs addition between A and B. The result
        of the addition is written to A.
        
        A = A + B
    
        A and B should have the same shapes (size).
        
        params:
            A: pointer to input matrix A
            B: pointer to input matrix B
            size: the length of A and B
    */

    
    //////////////////////////////////////////////////////////////////
    ////////////////////////// PROVIDED CODE /////////////////////////
    //////////////////////////////////////////////////////////////////
    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(x < size){
        A[x] = A[x] + B[x];
    }
    
    //////////////////////////////////////////////////////////////////
    ///////////////////////// END PROVIDED CODE //////////////////////
    //////////////////////////////////////////////////////////////////
}



__global__ void relu(float* input, int xLen, int yLen){

    /* 
        Calculates the nonlinear ReLU activation function.
        
        params:
            input: pointer to input matrix
            xLen: the size of the input in the x direction (number of columns)
            yLen: the size of the input in the y direction (number of rows)
    */
    
    //////////////////////////////////////////////////////////////////
    ////////////////////////// PROVIDED CODE /////////////////////////
    //////////////////////////////////////////////////////////////////
    
    // get thread indices
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    // if thread is in input block
    if(idx < xLen && idy < yLen){
    
        // 2D indices --> 1D index
        int index = idy * xLen + idx;

        // if element is negative replace value with zero
        if(input[index] < 0.0){
            input[index] = 0.0;
        }
    }
    
    //////////////////////////////////////////////////////////////////
    ///////////////////////// END PROVIDED CODE //////////////////////
    //////////////////////////////////////////////////////////////////
}


__global__ void conv2d(float* input, float* mask, 
                       float* output, int inputWidth,
                       int inputHeight, int maskSize){
    /* 
        Performs convolution between input and mask. 
        
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        
        We make the following assumptions:
            - the input is 2D (H, W) where H == W
            - mask is a 2D odd and square matrix (i.e. 3x3, 5x5, 7x7, etc.)
            - no padding is required
            - the output shape is the same as the input shape. 
              This corresponds to setting padding='same' in
              torch.nn.functional.conv2d. See the link below:
            - stride = 1
            - dilation = 1
            
        https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    
        params:
           input: pointer to the input matrix to be convolved
           mask: pointer to the convolutional mask matrix
           output: pointer to the output convolved matrix
           inputWidth: the input width in number of columns
           inputHeight: the input height in number of rows
           maskSize: the length of the square convolutional mask
    */
    
    //////////////////////////////////////////////////////////////////
    ////////////////////////// PROVIDED CODE /////////////////////////
    //////////////////////////////////////////////////////////////////
    
    // get thread indices
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    // if thread needs to calculate an output element
    if(idx < inputWidth && idy < inputHeight){
        
        // get mask offset start indices
        int inputStartX = idx - (maskSize / 2);
        int inputStartY = idy - (maskSize / 2);
        
        // iterate through kernel
        for(int j = 0; j < maskSize; j++){
        
            int currentY = inputStartY + j;
            
            for(int k = 0; k < maskSize; k++){
                
                int currentX = inputStartX + k;
                
                // if current indices are inside bounds, calculate output element and store value in output array
                if(currentY < inputHeight && currentY > -1 && currentX < inputWidth && currentX > -1){
                    output[idy * inputWidth + idx] += input[currentY * inputWidth + currentX] * mask[j * maskSize + k];
                }
            }
        }
    }
    
    //////////////////////////////////////////////////////////////////
    ///////////////////////// END PROVIDED CODE //////////////////////
    //////////////////////////////////////////////////////////////////
}


__global__ void MaxPool2d(float* input, float* output, int inputWidth, int inputHeight, int kernelSize, int stride){
    /* 
        Performs 2D max-pooling on the input array. 
        
        We make the following assumptions:
            - square pooling kernel
            - kernelSize == stride
            - square input array
                
        params:
           input: pointer to the input matrix to be max-pooled
           output: pointer to the output matrix, result of max-pooling
           inputWidth: the input width in number of columns
           inputHeight: the input height in number of rows
           kernelSize: the length of the square pooling window
           stride: the stride of the pooling operation. This is 
                   equal to kernelSize for our purposes.
    */
    
    //////////////////////////////////////////////////////////////////
    ////////////////////////// PROVIDED CODE /////////////////////////
    //////////////////////////////////////////////////////////////////
    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    int outputWidth = (inputWidth - kernelSize) / stride + 1;
    int outputHeight = (inputHeight - kernelSize) / stride + 1;
    
    int inputStartX = x * kernelSize;
    int inputStartY = y * kernelSize;
    
    float maxVal = -1000000000.0;
    
    if(x < outputWidth && y < outputHeight){
        for(int kernelRow = 0; kernelRow < kernelSize; kernelRow++){
            int windowX = inputStartX + kernelRow;
            for(int kernelCol = 0; kernelCol < kernelSize; kernelCol++){
                int windowY = inputStartY + kernelCol;
                float inputElement = input[windowY * inputWidth + windowX];
                if(inputElement > maxVal){
                    maxVal = inputElement;
                }
            }
        }
        output[y * outputWidth + x] = maxVal;
    }
    
    //////////////////////////////////////////////////////////////////
    ///////////////////////// END PROVIDED CODE //////////////////////
    //////////////////////////////////////////////////////////////////


__global__ void dot(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols){
    /* 
        Performs matrix multiplication between A and B. 
        
        A x B = C
        
        We make the following assumption:
            - ARows == BCols

        params:
           A: pointer to the input matrix A
           B: pointer to the output matrix B
           ARows: the number of rows in A
           ACols: the number of columns in A
           BRows: the number of rows in B
           BCols: the number of columns in B
           CRows: the number of rows in C
           CCols: the number of columns in C
    */
    //////////////////////////////////////////////////////////////////
    ////////////////////////// PROVIDED CODE /////////////////////////
    //////////////////////////////////////////////////////////////////
    
    float CValue = 0; // initialize value for C

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; // get thread indices
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; // allocate shared memory for A and B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // iterate through block
    for (int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; k++) {
        
        // if thread in block bounds, calculate matrix element output
        if(k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows){
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if(k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols){
            Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for(int n = 0; n < BLOCK_SIZE; ++n){ // write from shared memory to output
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }
             
         __syncthreads();
    }

    if (Row < CRows && Col < CCols){
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    }
        
    //////////////////////////////////////////////////////////////////
    ///////////////////////// END PROVIDED CODE //////////////////////
    //////////////////////////////////////////////////////////////////
}








