cpp = False

if cpp:
    try:
        import FEM2D._meshcpp as cpp_backend
    except ImportError as e:
        print(f"Could not import FEM2D._meshcpp: {e}")
        cpp_backend = None
else:
    cpp_backend = None