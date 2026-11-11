# Makes this directory a package, so pytest inserts the repository root
# (the first parent without __init__.py) into sys.path and the tests can
# ``import stockstats`` without a sys.path hack in conftest.py.
