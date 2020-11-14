@echo off
call python setup.py sdist bdist_wheel
call (echo TeamSqueem && makebigmap)| python -m twine upload dist/*