extern "C" __global__
void jacobi_smooth(
    float* __restrict__ u,
    const float* __restrict__ r_u,
    float kappa,
    float delta,
    float dx,
    int ny, int nx, 
    int stride, int halo, 
    float omega)
{

    int j = blockIdx.x * stride + (threadIdx.x - halo);
    int i = blockIdx.y * stride + (threadIdx.y - halo);

    if (i >= ny || j >= nx) return;

    bool is_active = (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo) &&
                     (threadIdx.y >= halo && threadIdx.y < blockDim.y - halo);

    if ( is_active) {
	float coeff_c = 4.0f;
	coeff_c += j < (nx-1) ? 0.0f : -1.0f;	
	coeff_c += j > 0      ? 0.0f : -1.0f;
	coeff_c += i > 0      ? 0.0f : -1.0f;
	coeff_c += i < (ny-1) ? 0.0f : -1.0f;

	float diagonal = delta * coeff_c / (dx * dx) + kappa * kappa;
	u[i * nx + j] -= omega * r_u[i * nx + j] / diagonal; 
    }
}
