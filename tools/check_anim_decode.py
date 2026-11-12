"""Offline checks of MKW 1.8.0 animation decoding against the user's NSO.

Usage: py -3 tools/check_anim_decode.py --exefs "C:/Games/MKW PC/exefs"

Executes only isolated ARM instruction blocks in Unicorn. Does not start the
game, load its data into a host executable, use a GPU, or change settings.
Requires the repository's existing lz4, capstone, and unicorn Python packages.
No extracted instructions, game data, or binary output are written or printed.

The numerical reference is an independent Python transcription of the current
runtime algorithm, not the compiled C helpers. These checks cannot establish
that the animation data or sample time supplied by the game is valid.
"""

import argparse
import collections
import json
import math
from pathlib import Path
import random
import re
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recomp"))

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from lifter import Lifter
from nso import Nso
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM
from unicorn import arm64_const as arm

BASE = 0x7000000000
FUNC_START, FUNC_END = 0x42BEA0, 0x42F2B0
CODE_PAGE = 0x42C000
HALF_START, HALF_END = 0x42C394, 0x42C404
FIELD_START, FIELD_END = 0x42C268, 0x42C294
TIME_START, TIME_CONVERTED, TIME_END = 0x42C130, 0x42C138, 0x42C148
HEADER, DATA, OUTPUT, FRAME = 0x100000, 0x101000, 0x102000, 0x103800
MASK32 = 0xFFFFFFFF


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def word_bytes(value):
    return struct.pack("<I", value)


def f32_from_bits(value):
    return struct.unpack("<f", word_bytes(value))[0]


def bits_from_f32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def read_word(uc, address):
    return struct.unpack("<I", uc.mem_read(address, 4))[0]


def set_registers(uc, **registers):
    for name, value in registers.items():
        uc.reg_write(getattr(arm, "UC_ARM64_REG_" + name.upper()), value)


def run_block(uc, start, end, maximum=100):
    uc.emu_start(BASE + start, BASE + end, count=maximum)
    require(uc.reg_read(arm.UC_ARM64_REG_PC) == BASE + end,
            "Unexpected control flow: supplied NSO may not be MKW 1.8.0")


def check_arithmetic(nso):
    """Check lifter operand fields against Capstone's independent decoder."""
    disassembler = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    lifter = Lifter(BASE, simd="c")
    binary_ops = {"fmul": "FOP_MUL", "fdiv": "FOP_DIV", "fadd": "FOP_ADD",
                  "fsub": "FOP_SUB", "fmaxnm": "FOP_MAXNM", "fnmul": "FOP_NMUL"}
    fma_signs = {"fmadd": (0, 0), "fmsub": (0, 1),
                 "fnmadd": (1, 1), "fnmsub": (1, 0)}
    counts = collections.Counter()
    instructions = disassembler.disasm(bytes(nso.image[FUNC_START:FUNC_END]),
                                      BASE + FUNC_START)
    for instruction in instructions:
        mnemonic = instruction.mnemonic
        if mnemonic not in binary_ops and mnemonic not in fma_signs:
            continue
        registers = re.findall(r"[sd](\d+)", instruction.op_str)
        precision = instruction.op_str[0]
        if mnemonic in binary_ops:
            expected = (f"fp_op_{precision}(c, {binary_ops[mnemonic]}, "
                        + ", ".join(registers) + ");")
        else:
            expected = (f"fp_fma_{precision}(c, "
                        + ", ".join(map(str, fma_signs[mnemonic])) + ", "
                        + ", ".join(registers) + ");")
        word = struct.unpack("<I", instruction.bytes)[0]
        actual = lifter.lift(word, instruction.address).code
        require(actual == expected, "Scalar operation/operand decode mismatch")
        counts[mnemonic] += 1
    require(sum(counts.values()) == 221,
            "Unexpected arithmetic instruction count: NSO version differs")
    return {"cases": sum(counts.values()), "mismatches": 0,
            "operations": dict(counts)}


def half_model(value):
    exponent, mantissa = (value >> 10) & 31, value & 1023
    if exponent:
        exponent += 0x70
    else:
        leading_zeros = 32 - mantissa.bit_length()
        exponent = 0 if not mantissa else (0x86 - leading_zeros) & MASK32
        shift = (10 - (leading_zeros ^ 31)) & 31
        mantissa = 0 if not mantissa else (value << shift) & 0x3FE
    return ((value << 16) & 0x80000000) | (mantissa << 13) | (exponent << 23)


def check_half(uc):
    uc.mem_write(HEADER + 0x114, bytes([16]) * 4)
    uc.mem_write(HEADER + 0x12A, bytes([2]))
    set_registers(uc, x8=HEADER, x9=4, x15=HEADER + 0x118, x20=0x86,
                  x21=10, x26=FRAME - 0x70, x25=FRAME - 0x4C,
                  x28=FRAME - 0x50, x23=FRAME - 0x54, x29=FRAME)
    finite_count = 0
    for value in range(65536):
        uc.mem_write(FRAME - 0x70, word_bytes(value))
        set_registers(uc, x11=0)
        run_block(uc, HALF_START, HALF_END)
        actual = read_word(uc, FRAME - 0x58)
        require(actual == half_model(value), "Half-decoder transcription mismatch")
        if ((value >> 10) & 31) < 31:
            independent = bits_from_f32(struct.unpack("<e", struct.pack("<H", value))[0])
            require(actual == independent, "Finite IEEE half conversion mismatch")
            finite_count += 1
    return {"cases": 65536, "mismatches": 0,
            "finite_ieee_cases": finite_count, "finite_ieee_mismatches": 0,
            "non_ieee_exponent31_cases": 2048}


def check_fields(uc, rng):
    words = [0, 1, MASK32, 0x80000000, 0x7FFFFFFF, 0xAAAAAAAA,
             0x55555555, 0x01234567] + [rng.getrandbits(32) for _ in range(16)]
    cases = independent_cases = 0
    for position in range(32):
        for width in range(33):
            for previous_word_done in (0, 1):
                for word in words:
                    uc.mem_write(HEADER + 1, bytes([width]))
                    uc.mem_write(DATA + previous_word_done * 4, word_bytes(word))
                    set_registers(uc, x10=HEADER, x11=DATA, x12=position,
                                  x13=previous_word_done, x14=1, x15=OUTPUT)
                    run_block(uc, FIELD_START, FIELD_END)
                    actual = read_word(uc, OUTPUT)
                    expected = ((word << position) & MASK32) >> ((-width) & 31)
                    end_position = position + width
                    next_position = 0 if end_position == 32 else end_position
                    require(actual == expected, "Packed field value mismatch")
                    require(uc.reg_read(arm.UC_ARM64_REG_X12) == next_position,
                            "Packed field bit-position mismatch")
                    require(uc.reg_read(arm.UC_ARM64_REG_X13) == (end_position == 32),
                            "Packed field boundary-flag mismatch")
                    require(uc.reg_read(arm.UC_ARM64_REG_X11) == DATA + previous_word_done * 4,
                            "Packed field source-pointer mismatch")
                    if 0 < width <= 32 - position:
                        independent = (word >> (32 - position - width)) & ((1 << width) - 1)
                        require(actual == independent, "Independent MSB field mismatch")
                        independent_cases += 1
                    cases += 1
    return {"cases": cases, "mismatches": 0,
            "independent_msb_cases": independent_cases}


def check_time(uc, rng):
    samples = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
               0x00800000, 0x80800000, 0x3F000000, 0xBF000000,
               0x3F800000, 0xBF800000, 0x3FC00000, 0xBFC00000,
               0x4EFFFFFF, 0x4F000000, 0xCF000000, 0xCF000001,
               0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001]
    samples.extend(rng.getrandbits(32) for _ in range(20000))
    for bits in samples:
        time = f32_from_bits(bits)
        if math.isnan(time):
            truncated = 0
        elif time >= 2147483648:
            truncated = 2147483647
        elif time < -2147483648:
            truncated = -2147483648
        else:
            truncated = math.trunc(time)
        converted_back = f32_from_bits(bits_from_f32(float(truncated)))
        correction = time != converted_back and (math.isnan(time) or time < 0)
        set_registers(uc, s0=bits, s1=0, s10=0)
        run_block(uc, TIME_START, TIME_CONVERTED)
        require(uc.reg_read(arm.UC_ARM64_REG_S1) == (truncated & MASK32),
                "Animation signed truncation mismatch")
        run_block(uc, TIME_CONVERTED, TIME_END)
        require(uc.reg_read(arm.UC_ARM64_REG_X9) == int(correction),
                "Animation negative-fraction correction mismatch")
    return {"cases": len(samples), "mismatches": 0, "fpcr": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exefs", type=Path, required=True,
                        help="User-supplied MKW 1.8.0 ExeFS directory containing main")
    args = parser.parse_args()
    nso = Nso(str(args.exefs / "main"))
    require(len(nso.image) > FUNC_END, "NSO image is too small")
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    uc.mem_map(BASE + CODE_PAGE, 0x1000)
    uc.mem_write(BASE + CODE_PAGE, bytes(nso.image[CODE_PAGE:CODE_PAGE + 0x1000]))
    uc.mem_map(HEADER, 0x4000)
    set_registers(uc, fpcr=0)
    rng = random.Random(0x42C268)
    results = {
        "scalar_arithmetic": check_arithmetic(nso),
        "half_conversion": check_half(uc),
        "packed_fields": check_fields(uc, rng),
        "animation_time": check_time(uc, rng),
        "limitations": [
            "Python models compare with isolated original ARM blocks, not compiled C helpers.",
            "Sample time adds +0; this checks conversion and FCCMP, not arbitrary input addition.",
            "FPCR is zero; other guest rounding/flush modes are outside this check.",
            "Exponent 31 follows the game's finite software representation, not IEEE NaN/Inf.",
            "Does not establish validity or lifetime of real animation curve data.",
        ],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
