// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }
};

// Used for vectorized allocations that are not in registers
template <typename scalar_t, int vec_size>
__device__ void arraySet(scalar_t* buff, scalar_t val) {
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    buff[i] = val;
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadGeneric(scalar_t* to, scalar_t* from) {
  // It would be really nice to use memcpy here, but one example was failing
  // with:
  //
  //  memcpy(to, from, vec_size * sizeof(scalar_t));
  //
  // Yet passing with:
  //
  // for(int i = 0; i < vec_size; i++){
  //   to[i] = from[i];
  // }

  switch (sizeof(scalar_t) * vec_size) {
    case 1:
      *reinterpret_cast<uchar1*>(to) = *reinterpret_cast<uchar1*>(from);
      break;
    case 2:
      *reinterpret_cast<uchar2*>(to) = *reinterpret_cast<uchar2*>(from);
      break;
    case 4:
      *reinterpret_cast<uint1*>(to) = *reinterpret_cast<uint1*>(from);
      break;
    case 8:
      *reinterpret_cast<uint2*>(to) = *reinterpret_cast<uint2*>(from);
      break;
    case 12:
      *reinterpret_cast<uint3*>(to) = *reinterpret_cast<uint3*>(from);
      break;
    case 16:
      *reinterpret_cast<uint4*>(to) = *reinterpret_cast<uint4*>(from);
      break;
  }
}

// Volatile version only works with c++ fundamnetal types
template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGenericVolatile(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 2:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 4:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 8:
      *reinterpret_cast<typename MaybeVolatile<double, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<double, is_volatile_from>::type*>(from);
      break;
  }
}

// template function specialization is not allowed, so we use a struct.
// loadLocalToGlobal and loadGlobalToLocal are structs with static call functions.
// array.cu does not need this, but hip-clang complains about discarding volatile if we don't.
// hip-clang doesn't optimize away if(is_volatile) when is_volatile is false.

template <typename scalar_t, int vec_size, bool is_volatile>
struct loadLocalToGlobal;

template <typename scalar_t, int vec_size>
struct loadLocalToGlobal<scalar_t, vec_size, true> {
static __device__ void call(
    typename MaybeVolatile<scalar_t, true>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, true, false>(to, from);
      break;
    case 8: {
        uint2 const& _from = *reinterpret_cast<uint2*>(from);
        volatile uint2 * _to = reinterpret_cast<volatile uint2*>(to);
        uint2 & __to = *const_cast<uint2*>(_to);
        __to = _from;
        // TODO memfence
      break;
    }
    case 12: {
        uint3 const& _from = *reinterpret_cast<uint3*>(from);
        volatile uint3 * _to = reinterpret_cast<volatile uint3*>(to);
        uint3 & __to = *const_cast<uint3*>(_to);
        __to = _from;
        // TODO memfence
      break;
    }
    case 16: {
        uint4 const& _from = *reinterpret_cast<uint4*>(from);
        volatile uint4 * _to = reinterpret_cast<volatile uint4*>(to);
        uint4 & __to = *const_cast<uint4*>(_to);
        __to = _from;
        // TODO memfence
      break;
    }
  }
}
};

template <typename scalar_t, int vec_size>
struct loadLocalToGlobal<scalar_t, vec_size, false> {
static __device__ void call(
    typename MaybeVolatile<scalar_t, false>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, false>(to, from);
      break;
    case 8: {
      uint2 const& _from = *reinterpret_cast<uint2*>(from);
      uint2 & _to = *reinterpret_cast<uint2*>(to);
      _to = _from;
      break;
    }
    case 12: {
      uint3 const& _from = *reinterpret_cast<uint3*>(from);
      uint3 & _to = *reinterpret_cast<uint3*>(to);
      _to = _from;
      break;
    }
    case 16: {
      uint4 const& _from = *reinterpret_cast<uint4*>(from);
      uint4 & _to = *reinterpret_cast<uint4*>(to);
      _to = _from;
      break;
    }
  }
}
};

template <typename scalar_t, int vec_size, bool is_volatile>
struct loadGlobalToLocal;

template <typename scalar_t, int vec_size>
struct loadGlobalToLocal<scalar_t, vec_size, true> {
static __device__ void call(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, true>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, true>(to, from);
      break;
    case 8: {
      uint2& _to = *reinterpret_cast<uint2*>(to);
      volatile uint2* _from = reinterpret_cast<volatile uint2*>(from);
      uint2& __from = *const_cast<uint2*>(_from);
      _to = __from;
      // TODO memfence
      break;
    }
    case 12: {
      uint3& _to = *reinterpret_cast<uint3*>(to);
      volatile uint3* _from = reinterpret_cast<volatile uint3*>(from);
      uint3& __from = *const_cast<uint3*>(_from);
      _to = __from;
      // TODO memfence
      break;
    }
    case 16: {
      uint4& _to = *reinterpret_cast<uint4*>(to);
      volatile uint4* _from = reinterpret_cast<volatile uint4*>(from);
      uint4& __from = *const_cast<uint4*>(_from);
      _to = __from;
      // TODO memfence
      break;
    }
  }
}
};

template <typename scalar_t, int vec_size>
struct loadGlobalToLocal<scalar_t, vec_size, false> {
static __device__ void call(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, false>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, false>(to, from);
      break;
    case 8: {
      uint2& _to = *reinterpret_cast<uint2*>(to);
      uint2& _from = *reinterpret_cast<uint2*>(from);
      _to = _from;
      break;
    }
    case 12: {
      uint3& _to = *reinterpret_cast<uint3*>(to);
      uint3& _from = *reinterpret_cast<uint3*>(from);
      _to = _from;
      break;
    }
    case 16: {
      uint4& _to = *reinterpret_cast<uint4*>(to);
      uint4& _from = *reinterpret_cast<uint4*>(from);
      _to = _from;
      break;
    }
  }
}
};

template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGlobalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
    case 2:
    case 4:
    case 8:
      loadGenericVolatile<scalar_t, vec_size, is_volatile_to, is_volatile_from>(
          to, from);
      break;
    case 12: {
      uint3 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>::call(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>::call(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
    case 16: {
      uint4 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>::call(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>::call(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
  }
}
