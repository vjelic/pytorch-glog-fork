#!/bin/bash

set -ex

patch /opt/rocm/include/rocprim/config.hpp <<'EOF'
--- config.hpp.orig	2025-04-02 18:23:59.000000000 +0000
+++ config.hpp.modified	2025-06-10 20:16:55.530550282 +0000
@@ -77,9 +77,12 @@
 #undef ROCPRIM_TARGET_CDNA1
 #undef ROCPRIM_TARGET_CDNA2
 #undef ROCPRIM_TARGET_CDNA3
+#undef ROCPRIM_TARGET_CDNA4
 
 // See https://llvm.org/docs/AMDGPUUsage.html#instructions
-#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
+#if defined(__gfx950__)
+    #define ROCPRIM_TARGET_CDNA4 1
+#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
     #define ROCPRIM_TARGET_CDNA3 1
 #elif defined(__gfx90a__)
     #define ROCPRIM_TARGET_CDNA2 1
EOF

patch /opt/rocm/include/rocprim/intrinsics/atomic.hpp <<'EOF'
--- intrinsics/atomic.hpp.orig	2025-04-02 18:23:59.000000000 +0000
+++ intrinsics/atomic.hpp.modified	2025-06-10 20:17:48.815358314 +0000
@@ -170,7 +170,7 @@
 #define ROCPRIM_ATOMIC_LOAD(inst, mod, wait, ptr) \
     asm volatile(inst " %0, %1 " mod "\t\n" wait : "=v"(result) : "v"(ptr) : "memory")
 
-#if ROCPRIM_TARGET_CDNA3
+#if ROCPRIM_TARGET_CDNA4 || ROCPRIM_TARGET_CDNA3
     #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
         ROCPRIM_ATOMIC_LOAD("flat_load_dwordx4", "sc1", "s_waitcnt vmcnt(0)", ptr)
     #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
@@ -280,7 +280,7 @@
 #define ROCPRIM_ATOMIC_STORE(inst, mod, wait, ptr) \
     asm volatile(inst " %0, %1 " mod "\t\n" wait : : "v"(ptr), "v"(value) : "memory")
 
-#if ROCPRIM_TARGET_CDNA3
+#if ROCPRIM_TARGET_CDNA4 || ROCPRIM_TARGET_CDNA3
     #define ROCPRIM_ATOMIC_STORE_FLAT(ptr) \
         ROCPRIM_ATOMIC_STORE("flat_store_dwordx4", "sc1", "s_waitcnt vmcnt(0)", ptr)
     #define ROCPRIM_ATOMIC_STORE_SHARED(ptr) \
EOF
