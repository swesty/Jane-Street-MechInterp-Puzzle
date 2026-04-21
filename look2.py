import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

print("Class module:", type(model).__module__)
print("Class name:", type(model).__name__)
print("MRO:", [c.__name__ for c in type(model).__mro__])
print("Number of children:", len(list(model.children())))

# look at child types
from collections import Counter
ctypes = Counter(type(c).__name__ for c in model.children())
print("Child types:", ctypes)

# check first child
first = list(model.children())[0]
print("First child class module:", type(first).__module__)
print("First child class:", type(first).__name__)

# try passing a string
print("\n--- Trying model('vegetable dog') ---")
try:
    out = model("vegetable dog")
    print("Output:", out)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# try a tensor of length 55
print("\n--- Trying model(torch.zeros(55)) ---")
try:
    out = model(torch.zeros(55))
    print("Output shape:", out.shape, "value:", out)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
