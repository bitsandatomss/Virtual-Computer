"""Root conftest: make layer packages importable for every test suite."""
import virtual_computer.paths  # noqa: F401  (inserts microarchitecture/compiler/kernel into sys.path)
