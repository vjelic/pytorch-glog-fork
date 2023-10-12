#ifndef INCBIN_H
#define INCBIN_H

#define STR2(x) #x
#define STR(x) STR2(x)

#ifdef _WIN32
#define INCBIN_SECTION ".rdata, \"dr\""
#else
#define INCBIN_SECTION ".rodata"
#endif

#define INCBIN_DEFINE(name, file, type) \
    __asm__(".section " INCBIN_SECTION "\n" \
            ".global " STR(name) "_start\n" \
            ".balign 16\n" \
            "" STR(name) "_start:\n" \
            ".incbin \"" file "\"\n" \
            \
            ".global " STR(name) "_end\n" \
            ".balign 1\n" \
            "" STR(name) "_end:\n" \
            ".byte 0\n" \
    )

#define INCBIN_DECLARE(name, file, type) \
    extern __attribute__((aligned(16))) const type name ## _start[]; \
    extern                              const type name ## _end[];   \
    extern                              const int  name ## _size = name ## _end - name ## _start

// this aligns start address to 16 and terminates byte array with explict 0
// which is not really needed, feel free to change it to whatever you want/need
#define INCBIN(name, file, type)    \
  INCBIN_DEFINE(name, file, type);  \
  INCBIN_DECLARE(name, file, type) 

#endif
