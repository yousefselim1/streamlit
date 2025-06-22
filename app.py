import streamlit as st
import torch
import torch.nn as nn

device = torch.device('cpu')

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim+num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        label_onehot = self.label_emb(labels)
        x = torch.cat([noise, label_onehot], dim=1)
        return self.model(x).view(-1, 1, 28, 28)

G = Generator().to(device)
G.load_state_dict(torch.load('generator_model.pth', map_location=device))
G.eval()

st.title('Handwritten Digit Image Generator')
digit = st.selectbox('Choose a digit (0-9):', list(range(10)))
if st.button('Generate Images'):
    noise = torch.randn(5, 100, device=device)
    labels = torch.full((5,), digit, dtype=torch.long, device=device)
    fake_imgs = G(noise, labels).detach().cpu().numpy()

    st.write(f'Generated images of digit {digit}')
    for i in range(5):
        img = fake_imgs[i].squeeze()
        img = (img + 1.0) / 2.0  # Scale to 0–1 range
        st.image(img, width=100, caption=f'Sample {i+1}')




