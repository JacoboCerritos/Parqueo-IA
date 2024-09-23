import torch

# Carga el modelo utilizando torch.jit.load()
model = torch.jit.load('Autos.pt', map_location=torch.device('cpu'))

# Guarda el modelo utilizando torch.jit.save()
torch.jit.save(model, 'Autos_yolo.pt')

# Carga el modelo
model = torch.jit.load('Autos_yolo.pt')
