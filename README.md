# Conditional VAE for MNIST

![ConditionalVAE](https://socialify.git.ci/Bhooyas/ConditionalVAE/image?font=KoHo&language=1&name=1&owner=1&pattern=Circuit%20Board&stargazers=1&theme=Auto)

A simple Conditional Variational Autoencoder written in PyTorch for MNIST dataset. Added a simple basic UI using Streamlit for it.

## Installation 

The first step would be to clone the project using the following command: - 
```
git clone https://github.com/Bhooyas/ConditionalVAE.git
```

All the configurations for the following are store in `config.py`. If required can be changed as desired.

The next step we go into the cloned directory and install the requirements using the follwoing command: - 
```
cd ConditionalVAE
pip install -r requirements.txt
```
**Note**: -  If you want ot directly run the inference skip the next two steps.

In the next step we train the model using `train.py` script. We use the `../data` directory for using MNSIT data. It will check the directory for data if it doesn't find will download the data at that location. You can change it if required.
```
python train.py
```

After running the above command, we will have a file named `MNIST_CVAE.safetensors`. These are the trained weights of the model. You can use this model weights if you want directly. In the next step we create a file named `mu_sigma.npz` which contains the min and max range of the mu and sigma.
```
python encoding.py
```

The next step is to run the streamlit UI and playaround with the model.
```
streamlit run app.py
```

## Next Steps
- [x] Train the model
- [x] Create a basic UI using Streamlit
- [x] Update the Readme
- [x] Convert the saved model to SafeTensor
- [ ] Add test based input for generation.