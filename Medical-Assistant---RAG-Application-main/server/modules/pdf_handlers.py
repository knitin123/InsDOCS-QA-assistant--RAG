import os
import shutil
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR=str(Path(__file__).resolve().parent.parent / "uploaded_docs")

def save_uploaded_files(files:list[UploadFile])-> list[str]:
    os.makedirs(UPLOAD_DIR,exist_ok=True)
    file_path=[]
    for file in files:
        # keep only the base name so a crafted filename can't escape UPLOAD_DIR
        temp_path=os.path.join(UPLOAD_DIR,Path(file.filename).name)
        with open(temp_path,"wb") as f:
            shutil.copyfileobj(file.file,f)
        file_path.append(temp_path)
    return file_path
