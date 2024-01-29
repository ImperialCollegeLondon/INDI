#go to the folder where montage will be saved
cd "$1"

montage 'results_a/*'$2'.png' 'results_b/*'$2'.png' -gravity Center -tile 6x3 -geometry 1000x1000+5+5 results_montage_$2.png