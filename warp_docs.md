10.24. Warp Matrix Functions
C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a warp. In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire warp, otherwise the code execution is likely to hang.

10.24.1. Description
All following functions and types are defined in the namespace nvcuda::wmma. Sub-byte operations are considered preview, i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This extra functionality is defined in the nvcuda::wmma::experimental namespace.

template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
fragment
An overloaded class containing a section of a matrix distributed across all threads in the warp. The mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures.

Only certain combinations of template arguments are allowed. The first template parameter specifies how the fragment will participate in the matrix operation. Acceptable values for Use are:

matrix_a when the fragment is used as the first multiplicand, A,

matrix_b when the fragment is used as the second multiplicand, B, or

accumulator when the fragment is used as the source or destination accumulators (C or D, respectively).

The m, n and k sizes describe the shape of the warp-wide matrix tiles participating in the multiply-accumulate operation. The dimension of each tile depends on its role. For matrix_a the tile takes dimension m x k; for matrix_b the dimension is k x n, and accumulator tiles are m x n.

The data type, T, may be double, float, __half, __nv_bfloat16, char, or unsigned char for multiplicands and double, float, int, or __half for accumulators. As documented in Element Types and Matrix Sizes, limited combinations of accumulator and multiplicand types are supported. The Layout parameter must be specified for matrix_a and matrix_b fragments. row_major or col_major indicate that elements within a matrix row or column are contiguous in memory, respectively. The Layout parameter for an accumulator matrix should retain the default value of void. A row or column layout is specified only when the accumulator is loaded or stored as described below.

load_matrix_sync
Waits until all warp lanes have arrived at load_matrix_sync and then loads the matrix fragment a from memory. mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. ldm describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for __half element type or multiple of 4 for float element type. (i.e., multiple of 16 bytes in both cases). If the fragment is an accumulator, the layout argument must be specified as either mem_row_major or mem_col_major. For matrix_a and matrix_b fragments, the layout is inferred from the fragment’s layout parameter. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp. This function must be called by all threads in the warp, or the result is undefined.

store_matrix_sync
Waits until all warp lanes have arrived at store_matrix_sync and then stores the matrix fragment a to memory. mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. ldm describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for __half element type or multiple of 4 for float element type. (i.e., multiple of 16 bytes in both cases). The layout of the output matrix must be specified as either mem_row_major or mem_col_major. The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp.

fill_fragment
Fill a matrix fragment with a constant value v. Because the mapping of matrix elements to each fragment is unspecified, this function is ordinarily called by all threads in the warp with a common value for v.

mma_sync
Waits until all warp lanes have arrived at mma_sync, and then performs the warp-synchronous matrix multiply-accumulate operation D=A*B+C. The in-place operation, C=A*B+C, is also supported. The value of satf and template parameters for each matrix fragment must be the same for all threads in the warp. Also, the template parameters m, n and k must match between fragments A, B, C and D. This function must be called by all threads in the warp, or the result is undefined.

If satf (saturate to finite value) mode is true, the following additional numerical properties apply for the destination accumulator:

If an element result is +Infinity, the corresponding accumulator will contain +MAX_NORM

If an element result is -Infinity, the corresponding accumulator will contain -MAX_NORM

If an element result is NaN, the corresponding accumulator will contain +0

Because the map of matrix elements into each thread’s fragment is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling store_matrix_sync. In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements, direct element access can be implemented using the following fragment class members.

enum fragment<Use, m, n, k, T, Layout>::num_elements;
T fragment<Use, m, n, k, T, Layout>::x[num_elements];
As an example, the following code scales an accumulator matrix tile by half.

wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
float alpha = 0.5f; // Same value for all threads in warp
/*...*/
for(int t=0; t<frag.num_elements; t++)
frag.x[t] *= alpha;
10.24.2. Alternate Floating Point
Tensor Cores support alternate types of floating point operations on devices with compute capability 8.0 and higher.

__nv_bfloat16
This data format is an alternate fp16 format that has the same range as f32 but reduced precision (7 bits). You can use this data format directly with the __nv_bfloat16 type available in cuda_bf16.h. Matrix fragments with __nv_bfloat16 data types are required to be composed with accumulators of float type. The shapes and operations supported are the same as with __half.

tf32
This data format is a special floating point format supported by Tensor Cores, with the same range as f32 and reduced precision (>=10 bits). The internal layout of this format is implementation defined. In order to use this floating point format with WMMA operations, the input matrices must be manually converted to tf32 precision.

To facilitate conversion, a new intrinsic __float_to_tf32 is provided. While the input and output arguments to the intrinsic are of float type, the output will be tf32 numerically. This new precision is intended to be used with Tensor Cores only, and if mixed with other floattype operations, the precision and range of the result will be undefined.

Once an input matrix (matrix_a or matrix_b) is converted to tf32 precision, the combination of a fragment with precision::tf32 precision, and a data type of float to load_matrix_sync will take advantage of this new capability. Both the accumulator fragments must have float data types. The only supported matrix size is 16x16x8 (m-n-k).

The elements of the fragment are represented as float, hence the mapping from element_type<T> to storage_element_type<T> is:

precision::tf32 -> float
10.24.3. Double Precision
Tensor Cores support double-precision floating point operations on devices with compute capability 8.0 and higher. To use this new functionality, a fragment with the double type must be used. The mma_sync operation will be performed with the .rn (rounds to nearest even) rounding modifier.

10.24.4. Sub-byte Operations
Sub-byte WMMA operations provide a way to access the low-precision capabilities of Tensor Cores. They are considered a preview feature i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This functionality is available via the nvcuda::wmma::experimental namespace:

namespace experimental {
    namespace precision {
        struct u4; // 4-bit unsigned
        struct s4; // 4-bit signed
        struct b1; // 1-bit
   }
    enum bmmaBitOp {
        bmmaBitOpXOR = 1, // compute_75 minimum
        bmmaBitOpAND = 2  // compute_80 minimum
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
}
For 4 bit precision, the APIs available remain the same, but you must specify experimental::precision::u4 or experimental::precision::s4 as the fragment data type. Since the elements of the fragment are packed together, num_storage_elements will be smaller than num_elements for that fragment. The num_elements variable for a sub-byte fragment, hence returns the number of elements of sub-byte type element_type<T>. This is true for single bit precision as well, in which case, the mapping from element_type<T> to storage_element_type<T> is as follows:

experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
experimental::precision::s4 -> int (8 elements in 1 storage element)
experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
T -> T  //all other types
The allowed layouts for sub-byte fragments is always row_major for matrix_a and col_major for matrix_b.

For sub-byte operations the value of ldm in load_matrix_sync should be a multiple of 32 for element type experimental::precision::u4 and experimental::precision::s4 or a multiple of 128 for element type experimental::precision::b1 (i.e., multiple of 16 bytes in both cases).

Note

Support for the following variants for MMA instructions is deprecated and will be removed in sm_90:

experimental::precision::u4

experimental::precision::s4

experimental::precision::b1 with bmmaBitOp set to bmmaBitOpXOR

bmma_sync
Waits until all warp lanes have executed bmma_sync, and then performs the warp-synchronous bit matrix multiply-accumulate operation D = (A op B) + C, where op consists of a logical operation bmmaBitOp followed by the accumulation defined by bmmaAccumulateOp. The available operations are:

bmmaBitOpXOR, a 128-bit XOR of a row in matrix_a with the 128-bit column of matrix_b

bmmaBitOpAND, a 128-bit AND of a row in matrix_a with the 128-bit column of matrix_b, available on devices with compute capability 8.0 and higher.

The accumulate op is always bmmaAccumulateOpPOPC which counts the number of set bits.

10.24.5. Restrictions
The special format required by tensor cores may be different for each major and minor device architecture. This is further complicated by threads holding only a fragment (opaque architecture-specific ABI data structure) of the overall matrix, with the developer not allowed to make assumptions on how the individual parameters are mapped to the registers participating in the matrix multiply-accumulate.

Since fragments are architecture-specific, it is unsafe to pass them from function A to function B if the functions have been compiled for different link-compatible architectures and linked together into the same device executable. In this case, the size and layout of the fragment will be specific to one architecture and using WMMA APIs in the other will lead to incorrect results or potentially, corruption.

An example of two link-compatible architectures, where the layout of the fragment differs, is sm_70 and sm_75.

fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }
// sm_70 fragment layout
$> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
// sm_75 fragment layout
$> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
// Linking the two together
$> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o
This undefined behavior might also be undetectable at compilation time and by tools at runtime, so extra care is needed to make sure the layout of the fragments is consistent. This linking hazard is most likely to appear when linking with a legacy library that is both built for a different link-compatible architecture and expecting to be passed a WMMA fragment.

Note that in the case of weak linkages (for example, a CUDA C++ inline function), the linker may choose any available function definition which may result in implicit passes between compilation units.

To avoid these sorts of problems, the matrix should always be stored out to memory for transit through external interfaces (e.g. wmma::store_matrix_sync(dst, …);) and then it can be safely passed to bar() as a pointer type [e.g. float *dst].

Note that since sm_70 can run on sm_75, the above example sm_75 code can be changed to sm_70 and correctly work on sm_75. However, it is recommended to have sm_75 native code in your application when linking with other sm_75 separately compiled binaries.

10.24.6. Element Types and Matrix Sizes
Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of matrix_a, matrix_b and accumulator matrix supported:

Matrix A

Matrix B

Accumulator

Matrix Size (m-n-k)

__half

__half

float

16x16x16

__half

__half

float

32x8x16

__half

__half

float

8x32x16

__half

__half

__half

16x16x16

__half

__half

__half

32x8x16

__half

__half

__half

8x32x16

unsigned char

unsigned char

int

16x16x16

unsigned char

unsigned char

int

32x8x16

unsigned char

unsigned char

int

8x32x16

signed char

signed char

int

16x16x16

signed char

signed char

int

32x8x16

signed char

signed char

int

8x32x16

Alternate Floating Point support:

Matrix A

Matrix B

Accumulator

Matrix Size (m-n-k)

__nv_bfloat16

__nv_bfloat16

float

16x16x16

__nv_bfloat16

__nv_bfloat16

float

32x8x16

__nv_bfloat16

__nv_bfloat16

float

8x32x16

precision::tf32

precision::tf32

float

16x16x8

Double Precision Support:

Matrix A

Matrix B

Accumulator

Matrix Size (m-n-k)

double

double

double

8x8x4

Experimental support for sub-byte operations:

Matrix A

Matrix B

Accumulator

Matrix Size (m-n-k)

precision::u4

precision::u4

int

8x8x32

precision::s4

precision::s4

int

8x8x32

precision::b1

precision::b1

int

8x8x128

10.24.7. Example
The following code implements a 16x16x16 matrix multiplication in a single warp.

#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

