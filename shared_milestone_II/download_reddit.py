import requests
import zipfile
import os


if __name__ == "__main__":

    download_url = "https://storage.googleapis.com/kaggle-data-sets/1137504/2993221/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240622%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240622T054802Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=48e5e892e8bb46b5fee2f3ebf836255ad4088feeeca5188d6400554fe6a6bc9626e1b4b1ee955e84e49951811d7ce48fa92692a6e961aa39d46257f97c2d3d96c096b01a5cbe0cefad9e89a247d40fdfa657747b0ff97c7bff8be75853320456cd7c923be2f8db10c7701e6bd85317e01880e17a573e995776c5c4ce35364ff436d8623d39fcd2ce0f0b2d30849532a4819fafc1396797ace06976a412f0752dee06ea76dd52d5535babf01a73d19ba9a182131f48760efc92bf906d95817c7e3245452da31bd18c2bdc53566521a59970a6378264d8992cbd1a8c29e6255c6dd760bed164793753bc47324f3a872b9b5f7209c2ec8021896b460f9a41e43b30"

    response = requests.get(download_url)

    zip_file_path = "../downloaded.zip"

    with open(zip_file_path, "wb") as file:
        file.write(response.content)

    extract_dir = "../reddit"

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    if os.path.exists(zip_file_path):
        os.remove(zip_file_path)