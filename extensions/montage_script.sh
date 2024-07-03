#go to the folder where montage will be saved
cd "$1"

montage 'results_a/*'$2'.png' 'results_b/*'$2'.png' -gravity Center -tile 6x3 -geometry 500x500+5+5 -depth 8 results_montage_$3_slice_$2.png

montage -pointsize 20 -label "$3_slice_$2" results_montage_$3_slice_$2.png -tile 1x1 -geometry +0+0 results_montage_$3_slice_$2.png