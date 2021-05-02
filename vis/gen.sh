for file in *.gv
do
  dot -Tpng "$file" -o "$file.png"
done

