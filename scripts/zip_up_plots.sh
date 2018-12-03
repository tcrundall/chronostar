dir_name="../${1}_plots"
rm -r $dir_name
mkdir $dir_name 
find . -name "*.pdf" -exec cp '{}' --parents $dir_name \;
zip -r "$dir_name.zip" $dir_name
