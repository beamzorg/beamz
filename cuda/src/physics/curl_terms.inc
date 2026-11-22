struct CurlTerm {int source_component;int derivative_axis;};
__host__ __device__ constexpr CurlTerm FirstCurlTerm(int component){return component==0?CurlTerm{2,1}:component==1?CurlTerm{0,0}:CurlTerm{1,2};}
__host__ __device__ constexpr CurlTerm SecondCurlTerm(int component){return component==0?CurlTerm{1,0}:component==1?CurlTerm{2,2}:CurlTerm{0,1};}
__host__ __device__ constexpr int CpmlAxis(int term){return term==0?1:(term==1?0:(term==2?0:(term==3?2:(term==4?2:(1)))));}
__host__ __device__ constexpr float CpmlSign(int term){return term%2==0?1.0f:-1.0f;}
