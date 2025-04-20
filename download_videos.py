import os
import tarfile
import gdown

extract_dir = "./videos"



# Primero recuperamos el archivo comprimido de Google Drive
# El ID del archivo se puede encontrar en la URL de Google Drive
file_id = "1ZO7nevZM0ym4Z30StoGnvJhTj0DdO4Jh"
url = f"https://drive.google.com/uc?id={file_id}"

output = "archive.tar.gz"

gdown.download(url, output, quiet=False)

os.makedirs(extract_dir, exist_ok=True)

# Ahora descomprimimos el archivo tar.gz
# Y lo guardamos en "./videos"
with tarfile.open(output, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print(f"File successfully extracted to: {extract_dir}")
