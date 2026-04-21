import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

print("=" * 60)
print("TYPE:", type(model).__name__)
print("=" * 60)
print(model)
print("=" * 60)
print("PARAMETERS:")
total = 0
for name, p in model.named_parameters():
    total += p.numel()
    print(f"  {name:60s} {tuple(p.shape)}  {p.dtype}")
print(f"TOTAL PARAMS: {total:,}")
print("=" * 60)
print("BUFFERS:")
for name, b in model.named_buffers():
    print(f"  {name:60s} {tuple(b.shape)}  {b.dtype}")
