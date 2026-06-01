# backend.py

USE_CPP = True

backend = None

if USE_CPP:
    try:
        import FEM2D._meshcpp as backend
    except ImportError:
        pass