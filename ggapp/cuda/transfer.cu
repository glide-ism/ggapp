
extern "C" __global__
void restrict_cell_avg(
    const float* f_fine,      // (ny_fine, nx_fine)
    float* f_coarse,          // (ny_coarse, nx_coarse)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        // Full-weighting (average 4 fine cells)
        f_coarse[idx] = 0.25f * (f_fine[2*J * (2*nx_coarse) + 2*I] +
                                 f_fine[2*J * (2*nx_coarse) + 2*I + 1] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I] +
                                 f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1]);
    }
}

extern "C" __global__
void restrict_cell_sum(
    const float* f_fine,      // (ny_fine, nx_fine)
    float* f_coarse,          // (ny_coarse, nx_coarse)
    const int ny_coarse,
    const int nx_coarse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_coarse * nx_coarse) {
        int J = idx / nx_coarse;
        int I = idx % nx_coarse;
        
        // Full-weighting (average 4 fine cells)
        f_coarse[idx] = (f_fine[2*J * (2*nx_coarse) + 2*I] +
                         f_fine[2*J * (2*nx_coarse) + 2*I + 1] +
                         f_fine[(2*J + 1) * (2*nx_coarse) + 2*I] +
                         f_fine[(2*J + 1) * (2*nx_coarse) + 2*I + 1]);
    }
}


extern "C" __global__
void prolongate_cell_injection(
    const float* h_coarse,    // (ny_coarse, nx_coarse)
    float* h_fine,            // (ny_fine, nx_fine)
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < ny_fine * nx_fine) {
        int j = idx / nx_fine;
        int i = idx % nx_fine;
        
        int J = j / 2;
        int I = i / 2;
        
        h_fine[idx] = h_coarse[J * (nx_fine/2) + I];
    }
}

extern "C" __global__
void prolongate_cell_bilinear(
    const float* H_coarse,
    float* H_fine,
    const int ny_fine,
    const int nx_fine)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny_fine * nx_fine;
    
    if (idx >= total) return;
    
    int j = idx / nx_fine;  // Fine grid row
    int i = idx % nx_fine;  // Fine grid col
    
    int ny_coarse = ny_fine / 2;
    int nx_coarse = nx_fine / 2;
    
    // Fine cell (j,i) center is at (j+0.5, i+0.5) in fine cell units
    // Coarse cell (J,I) center is at (2J+1, 2I+1) in fine cell units
    // Map fine position to coarse grid coordinates:
    //   J_float = (j + 0.5 - 1) / 2 = (j - 0.5) / 2
    float J_float = (j - 0.5f) * 0.5f;
    float I_float = (i - 0.5f) * 0.5f;
    
    // Clamp to valid interpolation region [0, n_coarse-1]
    J_float = fmaxf(0.0f, fminf(J_float, (float)(ny_coarse - 1)));
    I_float = fmaxf(0.0f, fminf(I_float, (float)(nx_coarse - 1)));
    
    // Integer indices for the 4 surrounding coarse cells
    int J_lo = (int)J_float;
    int I_lo = (int)I_float;
    int J_hi = min(J_lo + 1, ny_coarse - 1);
    int I_hi = min(I_lo + 1, nx_coarse - 1);
    
    // Fractional position within the coarse cell
    float t_y = J_float - J_lo;
    float t_x = I_float - I_lo;
    
    // Load the 4 coarse values
    float v00 = H_coarse[J_lo * nx_coarse + I_lo];
    float v01 = H_coarse[J_lo * nx_coarse + I_hi];
    float v10 = H_coarse[J_hi * nx_coarse + I_lo];
    float v11 = H_coarse[J_hi * nx_coarse + I_hi];
    
    // Bilinear interpolation
    H_fine[idx] = (1.0f - t_y) * ((1.0f - t_x) * v00 + t_x * v01)
                + t_y         * ((1.0f - t_x) * v10 + t_x * v11);
}

