"""Conservative source ALD/AST may-footprints for explicit Maxwell-v3 GS inputs.

This standalone tool does not change the compiler, service, cache or builder.
Input starts with the 0x50-byte SPH. --code-end is the exclusive instruction end
(C1 start); --c1-end is the exclusive constant-byte end. Neither bound is inferred.
All instruction slots are scanned, including predicated/unreachable words;
scheduler slots alone are excluded by position relative to the instruction start.

Field audit: local Ryujinx Decoders/InstTable.cs ALD/AST 13-bit prefixes;
InstDecoders.cs InstAld/InstAst/AlSize; Decoder.FillBlock scheduling;
Instructions/InstEmitAttribute.cs Ald/Ast/HasPrimitiveVertex/CanLoadOutput;
AttributeMap.HasPrimitiveVertex; OperandHelper.Register maps RZ to constant 0.
These describe the project's existing Maxwell decoder, not native hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
HEADER_SIZE = 0x50
ALL = [0xffffffff] * 8
OPCODE_MASK = 0xfff8000000000000
ALD = 0xefd8000000000000
AST = 0xeff0000000000000
SPACES = ('input_read', 'output_read', 'output_write')


def analyze(data, *, code_end, c1_end, context=''):
    """Return source may-access masks; unknown addressing saturates its spaces.

    This does not execute CFG/register dataflow or infer definite reads/writes.
    In particular, a register selector remains dynamic after an earlier MOVI.
    Per-patch has a different namespace: its all-required mask is a rejection
    guard, not a normal-attribute interpretation of that namespace.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError('source must be bytes')
    if any(type(n) is not int for n in (code_end, c1_end)):
        raise ValueError('source bounds must be integers')
    if not HEADER_SIZE + 16 <= code_end <= c1_end <= len(data):
        raise ValueError('require SPH, scheduler, instruction and bounded C1 end')
    # Instructions are 64-bit words. The separate C1 payload is opaque bytes,
    # as in Program.cs Accessor: short tails are valid and never scanned.
    if (code_end - HEADER_SIZE) % 8:
        raise ValueError('instruction end must preserve whole 64-bit words')
    if (code_end - HEADER_SIZE) % 32 == 8:
        raise ValueError('instruction region ends with an empty scheduler bundle')
    if not isinstance(context, str) or len(context) > 128:
        raise ValueError('context must be a string of at most 128 characters')
    word0 = struct.unpack_from('<I', data)[0]
    if (word0 & 31, word0 >> 5 & 31, word0 >> 10 & 15) != (1, 3, 4):
        raise ValueError('only explicit Maxwell kind 1/version 3 geometry SPH is supported')
    masks = {name: [0] * 8 for name in SPACES}
    unknown = {name: False for name in SPACES}
    counts = dict(scheduler_words=0, other_words=0, ald=0, ast=0, predicated_attribute_sites=0,
                  per_patch_sites=0, output_read_sites=0, physical_address_sites=0,
                  dynamic_attribute_address_sites=0, dynamic_vertex_selector_sites=0,
                  zero_vertex_selector_sites=0, non_vertex_selector_sites=0,
                  misaligned_address_sites=0, out_of_range_address_sites=0,
                  unsupported_auxiliary_selector_sites=0, unsupported_register_span_sites=0,
                  unrecognized_mode_sites=0)
    reasons = set()

    def unknown_space(space, reason):
        unknown[space] = True
        masks[space] = ALL.copy()
        reasons.add(reason)

    for offset in range(HEADER_SIZE, code_end, 8):
        if (offset - HEADER_SIZE) % 32 == 0:
            counts['scheduler_words'] += 1
            continue
        word = struct.unpack_from('<Q', data, offset)[0]
        opcode = word & OPCODE_MASK
        if opcode not in (ALD, AST):
            counts['other_words'] += 1
            continue
        load = opcode == ALD
        counts['ald' if load else 'ast'] += 1
        address, words = word >> 20 & 0x7ff, (word >> 47 & 3) + 1
        ra, selector = word >> 8 & 255, word >> 39 & 255
        per_patch = bool(word >> 31 & 1)
        output_read = load and bool(word >> 32 & 1)
        space = 'output_read' if output_read else 'input_read' if load else 'output_write'
        # Unnamed fields are not evidence for a hardware address mode. Reject
        # their precision even though the pinned decoder does not consume them.
        known_fields = OPCODE_MASK | ((1 << (33 if load else 32)) - 1) | (0x3ff << 39)
        unrecognized_mode = bool(word & ~known_fields)
        counts['unrecognized_mode_sites'] += unrecognized_mode
        counts['predicated_attribute_sites'] += (word >> 16 & 7) != 7 or bool(word >> 19 & 1)
        counts['per_patch_sites'] += per_patch
        counts['output_read_sites'] += output_read
        physical = not per_patch and address == 0 and ra != 255
        counts['physical_address_sites'] += physical
        # Even a non-physical Ra mode is not treated as direct: the reference
        # has a TODO for indirect AST. Its ignored Ra is no address proof.
        dynamic_address = ra != 255
        counts['dynamic_attribute_address_sites'] += dynamic_address
        has_vertex = load and not per_patch and not output_read and (physical or address not in (0x60, 0x2f0, 0x2f4))
        if has_vertex:
            counts['zero_vertex_selector_sites' if selector == 255 else 'dynamic_vertex_selector_sites'] += 1
        else:
            counts['non_vertex_selector_sites'] += 1
        bad_aux = not load and selector != 255
        counts['unsupported_auxiliary_selector_sites'] += bad_aux
        bad_register = (word & 255) + words > 256
        counts['unsupported_register_span_sites'] += bad_register
        if unrecognized_mode:
            unknown_space(space, 'mode bits outside the audited decoder fields')
            if output_read:
                unknown_space('input_read', 'unrecognized ALD.O mode cannot exclude input reads')
        if per_patch:
            unknown_space(space, 'per-patch namespace is unsupported')
        if dynamic_address:
            unknown_space(space, 'physical or register-based attribute address')
            if output_read:
                unknown_space('input_read', 'dynamic ALD.O may select input-only tessellation fallback')
        if bad_aux:
            unknown_space(space, 'AST auxiliary selector is outside the direct GS subset')
        if bad_register:
            unknown_space(space, 'attribute data-register span exceeds RZ')
        if address % 4:
            counts['misaligned_address_sites'] += 1
            unknown_space(space, 'attribute byte address is not component-aligned')
        if address + words * 4 > 0x400:
            counts['out_of_range_address_sites'] += 1
            unknown_space(space, 'attribute span exceeds 256-component mask')
        if unrecognized_mode or per_patch or dynamic_address or bad_aux or bad_register or address % 4 or address + words * 4 > 0x400:
            continue
        for component in range(address // 4, address // 4 + words):
            target = space
            if output_read and component in (0x2f0 // 4, 0x2f4 // 4):
                # InstEmit.Ald's CanLoadOutput falls back to input for these
                # non-GS attributes. Preserve a conservative rejection guard.
                unknown_space('input_read', 'ALD.O tessellation fallback is outside GS scope')
                unknown_space('output_read', 'ALD.O tessellation fallback is outside GS scope')
            masks[target][component // 32] |= 1 << (component % 32)

    def projection(mask):
        users = [(mask[1 + location // 8] >> (location % 8 * 4)) & 15 for location in range(32)]
        return dict(position_components=mask[0] >> 28 & 15, user_components=users,
                    user_locations=sum(1 << i for i, nibble in enumerate(users) if nibble))

    return dict(schema='maxwell-gs-ald-ast-source-may', schema_version=1, context=context,
                source_sha256=hashlib.sha256(data).hexdigest(),
                declared_source_sha256=hashlib.sha256(data[:c1_end]).hexdigest(),
                instruction_region_sha256=hashlib.sha256(data[HEADER_SIZE:code_end]).hexdigest(),
                bounds=dict(file_bytes=len(data), header_bytes=HEADER_SIZE, code_end=code_end, c1_end=c1_end),
                source_context=dict(kind=1, version=3, stage='gs', passthrough=bool(word0 >> 24 & 1),
                                    scheduler='one 64-bit scheduler at each 32-byte instruction-relative boundary'),
                counts=counts, masks=masks, unknown=unknown,
                projections={name: projection(mask) for name, mask in masks.items()},
                unsupported=bool(reasons), unsupported_reasons=sorted(reasons),
                ald_ast_address_coverage_known=not any(unknown.values()), complete_shader_io=False,
                limitations=['Source ALD/AST may-accesses only; other instructions are counted, not decoded.',
                             'No reachability, predicate execution, register dataflow or definite-access proof.',
                             'No native-v4, S2R, synthesized epilogue, shader linkage or hardware equivalence proof.'])


def output_path(path):
    resolved = path.resolve()
    build = (ROOT / 'build').resolve()
    if resolved == build or not resolved.is_relative_to(build):
        raise ValueError('derived JSON must be below this checkout build directory')
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--code-end', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--c1-end', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--context', default='')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = output_path(args.out)
        report = analyze(args.code.read_bytes(), code_end=args.code_end, c1_end=args.c1_end, context=args.context)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing any existing result or source.
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Wrote source ALD/AST may-footprint; complete shader I/O is not established.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
