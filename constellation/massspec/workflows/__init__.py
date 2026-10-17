"""Multi-step mass-spec workflows: the code between a command and the pieces it drives.

A workflow owns a directory layout and the order things happen in
(convert, register, write, run, parse, record). The pieces it calls are
ordinary library functions that know nothing about the layout.

    encyclopedia    one acquisition searched with EncyclopeDIA, written
                    as an experiment directory
"""
