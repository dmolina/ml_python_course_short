cd presentation && pandoc presentation.md -t revealjs -s -Vtheme:sky -o presentation.html && cd -

for file in source/day1/0*.md; do
    fname=$(basename $file)
    foutput=notebooks/day1/${fname%.*}.ipynb

    if [ -f $foutput ]; then
        jupytext $file --to ipynb --update -o $foutput
    else
        jupytext $file --to ipynb -o $foutput
    fi
done

cp source/day1/*.csv notebooks/day1/
cp source/day1/*.arff notebooks/day1/
cp source/day2/*.csv notebooks/day2/

jupyter notebook notebooks
