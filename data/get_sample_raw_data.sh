BASE_DIR=LM_2D_CARE/raw
RAW_ARCHIVE=Denoising_Planaria.tar.gz

date
echo ; echo "Downloading data archive..." ;
wget -q --show-progress https://edmond.mpg.de/api/access/datafile/264085 -O $RAW_ARCHIVE

date
echo ; echo "Extracting data..." ;
mkdir -p $BASE_DIR
tar -xzf $RAW_ARCHIVE --strip-components=1 -C $BASE_DIR

date
echo ; echo "Cleaning..." ;
rm $RAW_ARCHIVE
rm $BASE_DIR/poster.png $BASE_DIR/Readme.md
rm -r $BASE_DIR/train_data
mv $BASE_DIR/test_data/* $BASE_DIR
rm -r $BASE_DIR/test_data

date
