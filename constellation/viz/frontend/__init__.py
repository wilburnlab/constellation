"""Frontend source tree.

The TS source under `src/` is committed; the built bundle under
`constellation/viz/static/<entry>/` is git-ignored and produced by
`python -m constellation.viz.frontend.build`. CI builds the bundle on
tag push and ships it inside release wheels.
"""
