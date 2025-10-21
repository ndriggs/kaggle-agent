import os
import glob
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_and_unzip_competition(competition: str, download_path: str = '.') -> None:
	api = KaggleApi()
	api.authenticate()

	print(f"Downloading competition files for '{competition}' to '{download_path}'...")
	# This will download a zip (or multiple files) into download_path
	api.competition_download_files(competition, path=download_path, quiet=False)

	# Try to find zip files that match the competition name first, then any recent zip
	pattern = os.path.join(download_path, f"{competition}*.zip")
	candidates = glob.glob(pattern)
	if not candidates:
		candidates = glob.glob(os.path.join(download_path, "*.zip"))

	if not candidates:
		print("No .zip files found after download. If the Kaggle API returned individual files instead of a zip, check the download_path for files.")
		return

	# Choose the most recently modified zip file
	zip_path = max(candidates, key=os.path.getmtime)
	extract_dir = os.path.join(download_path, competition)
	os.makedirs(extract_dir, exist_ok=True)

	print(f"Found zip file to extract: {zip_path}\nExtracting to: {extract_dir}")
	try:
		with zipfile.ZipFile(zip_path, 'r') as zf:
			zf.extractall(extract_dir)
		print("Extraction complete.")
	except zipfile.BadZipFile:
		print(f"Error: '{zip_path}' is not a valid zip file.")
	except Exception as e:
		print(f"Extraction failed: {e}")


if __name__ == '__main__':
	COMPETITION = 'home-data-for-ml-course'
	DOWNLOAD_PATH = '.'
	download_and_unzip_competition(COMPETITION, DOWNLOAD_PATH)