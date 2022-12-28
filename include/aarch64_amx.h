// AMX instructions are of the form:
//
//   0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)
//
// AMX must be explicitly enabled using op=17, operand=0 and disabled using
// op=17, operand=1. In Accelerate, these instructions are always prefixed
// by three nops. What could go wrong?
//
// If instructions other than "enable" are executed when AMX is not enabled,
// they are treated as illegal instructions.
//
//
// All other operations (op=0-16 and op=18-22) seem to take a 64-bit register
// number (X0-X30 or 31=XZR) as the operand.
//
// This register is typically a bitfield containing further parameters to the
// operation. For example, loads and stores have a 56-bit address in bits 0
// through 55, a 5-bit register offset (in units of 0x40) in bits 56
// through 61, and a 1-bit flag in bit 62 (acting as an 0x40 byte load/store
// when zero, or an 0x80 byte (but aligned) load/store when one).
//


#pragma once
#include <stdint.h>

#define PTR_ROW_FLAGS(ptr, row, flags) \
  (((uint64_t) & *(ptr)) + (((uint64_t)((row) + (flags)*64)) << 56))


//In Apple's Accelerate, instruction 17 is apparently always prefixed by three nops. 
#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)    AMX_OP_GPR( 0, gpr)
#define AMX_LDY(gpr)    AMX_OP_GPR( 1, gpr)
#define AMX_STX(gpr)    AMX_OP_GPR( 2, gpr)
#define AMX_STY(gpr)    AMX_OP_GPR( 3, gpr)
#define AMX_LDZ(gpr)    AMX_OP_GPR( 4, gpr)
#define AMX_STZ(gpr)    AMX_OP_GPR( 5, gpr)
#define AMX_LDZI(gpr)   AMX_OP_GPR( 6, gpr)
#define AMX_STZI(gpr)   AMX_OP_GPR( 7, gpr)
#define AMX_EXTRX(gpr)  AMX_OP_GPR( 8, gpr)
#define AMX_EXTRY(gpr)  AMX_OP_GPR( 9, gpr)
#define AMX_FMA64(gpr)  AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr)  AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr)  AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr)  AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr)  AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr)  AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr)  AMX_OP_GPR(16, gpr)
#define AMX_SET()       AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()       AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr)  AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr)  AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)
