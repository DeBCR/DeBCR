
BASE_DIR=LM_2D_CARE/npz

BASE_LINK=https://zenodo.org/records/12626122/files/LM_2D_CARE

mkdir -p $BASE_DIR

for SUBSET in train val test ; do
	echo ; echo "Downloading sample "$SUBSET" data..." ; 
	SUBSET_DIR=$BASE_DIR/$SUBSET
	mkdir -p $SUBSET_DIR
	wget -q --show-progress ${BASE_LINK}_${SUBSET}.npz -P $SUBSET_DIR
done
