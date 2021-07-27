for file in *.gv
do
  dot -Tpng "$file" -o "i$file.png"
done

