#!/bin/bash

path=~/as/DS_10283_853

file=$1
set=(${file//./ })
 dirname=$path/$set
mkdir -p $dirname


cat $path/$1 | while read line
do
	wavepath=$path/wav/$line
	array=(${line//// })
	# echo ${array[0]}
	dir=$dirname/${array[0]}
	# echo dirname is $dir
	mkdir -p $dir

	mv $wavepath $dir
 # 	if [ ! -d $dir ];then
	# 	mkdir $dir
	# else
	# 	echo dir exist
	# fi
	
	# echo $wavepath


	# array=(${line//// })
	# echo ${array[0]}
done

# echo $files11
# for element in $files
# do
# 	#echo $element
#       dir_new=$path"/"$element
#       mkdir $dir_new/wav1
#       mv $dir_new/*.wav $dir_new/wav1


# done    
