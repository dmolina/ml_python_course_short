cd presentation && pandoc intro_en.md -t revealjs -s -Vtheme:sky -o intro_en.html && cd -
jupytext sesion_1_intro_python_en.py --to ipynb
jupyter notebook
