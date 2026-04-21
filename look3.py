import torch
import torch.nn as nn

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

first = model[0]
print("=== first layer ===")
print("type:", type(first), "module:", type(first).__module__)
print("dir extras:", [x for x in dir(first) if not x.startswith('_')][:30])
print("attrs:", {k: type(v).__name__ for k, v in first.__dict__.items() if not k.startswith('_')})

# inspect __dict__
for k, v in first.__dict__.items():
    print(f"  {k!r}: {type(v).__name__}")

# Is there a forward hook?
print("\n_forward_pre_hooks:", first._forward_pre_hooks)
print("_forward_hooks:", first._forward_hooks)

# Check if the model itself has a custom forward
print("\nmodel.__dict__ keys:", list(model.__dict__.keys()))
print("model._forward_pre_hooks:", model._forward_pre_hooks)
print("model._forward_hooks:", model._forward_hooks)

# Try inspecting the first 5 children more carefully
for i, child in enumerate(list(model.children())[:5]):
    print(f"\nchild {i}: {type(child).__name__} class id={id(type(child))}")
    print(f"  vars: {list(vars(child).keys())}")
