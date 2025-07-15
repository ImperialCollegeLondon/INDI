# go to the folder where montage will be saved
Set-Location $args[0]

$slice_num = '{0:d2}' -f $args[1]
$folder_id = $folder_id

magick montage  ('results_a\*' + $slice_num + '.png') ('results_b\*' + $slice_num + '.png') -gravity Center -tile 9x3 -geometry 500x500+5+5 -depth 8 ("results_montage_" + $folder_id + "_slice_" + $slice_num + ".png")

magick montage -pointsize 20 -label ($folder_id+"_slice_$($slice_num)") ("results_montage_"+$folder_id+"_slice_$($slice_num).png") -tile 1x1 -geometry +0+0 ("results_montage_" + $folder_id + "_slice_$($slice_num).png")