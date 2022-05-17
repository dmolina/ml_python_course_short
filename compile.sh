cd presentation && pandoc intro_en.md -t revealjs -s -Vtheme:sky -o intro_en.html && cd -

for file in source/day1/*.md; do
    fname=$(basename $file)
    jupytext $file --to ipynb -o notebooks/day1/${fname%.*}.ipynb
done

cp source/day1/*.csv notebooks/day1/
cp source/day2/*.csv notebooks/day2/

jupyter notebook notebooks
