"""HPLC raw-data readers; subclass core.io.RawReader and register.

Modules (TODO):
    agilent_dx       - OpenLab .dx (OPC zip; Signal179 / InstrumentTrace179
                       / Spectra131 decoders). Direct torch.frombuffer for
                       the little-endian payloads; no numpy intermediate.
"""
