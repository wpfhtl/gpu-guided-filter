
__device__ void compute_cov_var(float4 *mean_Ip, float4 *mean_II, float4 *mean_I,
        float4 *mean_p, float4 *var_I, float4 *cov_Ip, int width, int height);

__device__ void compute_ab(float4 *var_I, float4 *cov_Ip, float4 *mean_I,
        float4 *mean_p, float4 *a, float4 *b, float eps, int width, int height);

__device__ void compute_q(float4 *in, float4 *mean_a, float4 *mean_b, float4 *q,
        int width, int height);

__global__ void mean_kernel(float4* d_input,
        float4 *d_p,
        float4 *mean_I,
        float4 *mean_p,
        float4 *mean_Ip,
        float4 *mean_II,
        float4 *d_tmp,
        float4 *d_tmp2,
        int width, int height,
        float eps);

__global__ void cov_var_ab_kernel(float4* d_input,
        float4 *mean_I,
        float4 *mean_p,
        float4 *mean_Ip,
        float4 *mean_II,
        float4 *var_I,
        float4 *cov_Ip,
        float4 *a, 
        float4 *b,
        float4 *mean_a,
        float4 *mean_b,
        int width, int height,
        float eps);

__global__ void output_kernel(float4* d_input,
        float4 *d_p,
        float4 *d_q,
        float4 *a, 
        float4 *b,
        float4 *mean_a,
        float4 *mean_b,
        int width, int height,
        float eps)


// #define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line);

void guided_filter_cuda(float4 *h_input,
        float4 *h_p,
        float4 *h_output,
        float4 *h_tmp,
        float4 *h_tmp2,
        int width, int height,
        float eps);

void compute(std::string input_file, std::string g_file, std::string output_file);

int main(int argc, char *argv[]);