extern "C" __global__
void compute_residual(
    float* __restrict__ r_u,
    const float* __restrict__ u,
    const float* __restrict__ rhs,
    float kappa,
    float delta,
    float dx,
    int ny, int nx, 
    int stride, int halo)
{
    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i >= ny || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active) {
	float u_c = u[i * nx + j];
	float u_r = j < (nx-1) ? u[i * nx + j + 1] : u[i * nx + j];	
	float u_l = j > 0      ? u[i * nx + j - 1] : u[i * nx + j];
	float u_t = i > 0      ? u[(i - 1) * nx + j] : u[i * nx + j];
	float u_b = i < (ny-1) ? u[(i + 1) * nx + j] : u[i * nx + j];

        
        float laplacian = delta*(4.0f * u_c - u_r - u_l - u_t - u_b)/(dx * dx);
        float mass = kappa * kappa * u_c;
	float forcing = rhs[i * nx + j];

        r_u[i * nx + j] = laplacian + mass - forcing; 

    }
}
