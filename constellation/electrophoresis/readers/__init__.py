"""Capillary-electrophoresis raw-data readers; subclass core.io.RawReader.

Modules (TODO):
    agilent_fa       - .raw decoder for FA + PA cartridges. Big-endian
                       uint16 frames; per-capillary trace = 5-pixel sum
                       around cap_pos[k] + 4 from the 0x3E8 table.
                       RunGeometry parameterises N_CAPILLARIES /
                       FRAME_BYTES so PA support is constants-only.
"""
