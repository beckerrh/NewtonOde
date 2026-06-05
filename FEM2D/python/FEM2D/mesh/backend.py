cpp = True

if cpp:
    try:
        import FEM2D._meshcpp as backend
    except ImportError as e:
        print(f"Could not import FEM2D._meshcpp: {e}")
        backend = None
else:
    backend = None