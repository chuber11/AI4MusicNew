
python create_verification_video.py data/train/Hands_Across_the_Sea data/train/Hands_Across_the_Sea/video_oracle.mp4 --ann-file data/train/Hands_Across_the_Sea/annotations_20260321_182747.json
python create_verification_video.py data/dev/Barnum_and_Baileys_Favorite data/dev/Barnum_and_Baileys_Favorite/video_oracle.mp4 --ann-file data/dev/Barnum_and_Baileys_Favorite/annotations_20260327_113038.json

python create_verification_video.py data/train/Hands_Across_the_Sea data/train/Hands_Across_the_Sea/video_model.mp4 --ann-file data/train/Hands_Across_the_Sea/predictions.json
#python create_verification_video.py data/dev/Barnum_and_Baileys_Favorite data/dev/Barnum_and_Baileys_Favorite/video_model.mp4 --ann-file data/dev/Barnum_and_Baileys_Favorite/predictions.json
