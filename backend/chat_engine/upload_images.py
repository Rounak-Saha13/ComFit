import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=r'D:\Sahithi\9_3_2025_ComFit\.env')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

bucket_name = 'comfit_images'
images_folder = r'D:\Sahithi\9_3_2025_ComFit\ComFit\extracted_images_for_upload'

def upload_all_images(folder):
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return
    
    files = os.listdir(folder)
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as f:
                print(f"Uploading {file_name}...")
                response = supabase.storage.from_(bucket_name).upload(file_name, f)
                print("Response:", response)
    print("All images uploaded.")

if __name__ == "__main__":
    upload_all_images(images_folder)
