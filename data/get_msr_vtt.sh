# This script provides instructions for downloading the MSR-VTT Dataset.

# Create directories for the raw data
mkdir -p data/raw_videos
mkdir -p data/annotations

echo "--- MSR-VTT Dataset Download Guide ---"
echo ""
echo "Please follow these manual steps:"
echo ""
echo "1. Go to the official dataset download page:"
echo "   https://www.robots.ox.ac.uk/~maxbain/video-mscoco/"
echo ""
echo "2. Download the following files:"
echo "   - Video Files: 'MSRVTT.zip' (This is a large file, ~11GB)"
echo "   - Annotations: 'MSRVTT-QA.zip' and 'videotranslate.zip' (or 'train_val_videodatainfo.json')"
echo ""
echo "3. Extract the 'MSRVTT.zip' archive. It will contain a 'TrainValVideo' and a 'TestVideo' folder."
echo "   Move all the .mp4 video files from both folders into 'data/raw_videos/'"
echo ""
echo "4. Extract the annotation zip files and move the relevant JSON/TXT files"
echo "   (e.g., 'MSRVTT_QA_train.json', 'videodatainfo.json') into the 'data/annotations/' directory."
echo ""
echo "After these files are in place, you can proceed with training the models."
echo ""
echo "------------------------------------------"
