import gdown

# Google Drive file ID
file_id = "1Di6WhpYwRSgYV_zpCOZniuO20WFDHSHb"

# File download location
output = "fraudTrain.csv"

# Download from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
print("Download complete: fraudTrain.csv")
