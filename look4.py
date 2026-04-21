import torch
import inspect as ins
import dis

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

ci = model.__dict__["_call_impl"]
print("type:", type(ci))
print("repr:", ci)
print()

# If it's a function/method, inspect it
if hasattr(ci, "__code__"):
    code = ci.__code__
    print("co_varnames:", code.co_varnames)
    print("co_freevars:", code.co_freevars)
    print("co_cellvars:", code.co_cellvars)
    print("co_consts:", code.co_consts)
    print("co_names:", code.co_names)
    print()
    print("=== source try ===")
    try:
        print(ins.getsource(ci))
    except Exception as e:
        print(f"no source: {e}")
    print()
    print("=== bytecode ===")
    dis.dis(ci)

# closure?
if hasattr(ci, "__closure__") and ci.__closure__:
    print("\n=== closure cells ===")
    for i, cell in enumerate(ci.__closure__):
        try:
            v = cell.cell_contents
            print(f"  [{i}] {type(v).__name__}: {repr(v)[:200]}")
        except ValueError:
            print(f"  [{i}] empty")
