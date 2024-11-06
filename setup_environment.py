# setup_environment.py

def setup_libraries():
    import os
    print("Installing pytorch_lightning...")
    os.system('pip install pytorch_lightning')

    print("Installing pydicom...")
    os.system('pip install pydicom')

    print("Installing opensimplex...")
    os.system('pip install opensimplex')

    print("Installing click, requests, tqdm, pyspng, ninja, imageio-ffmpeg...")
    os.system('pip install click requests tqdm pyspng ninja imageio-ffmpeg')

    print("Installing piq...")
    os.system('pip install piq')

    print("Installing openai-clip")
    os.system('pip install openai-clip')

    print("All packages have been installed.")

