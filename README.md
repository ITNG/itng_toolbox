### This is ITNG toolbox package.
-  IASBS Theoretical Neuroscience Group toolbox, to analysis the time series, spike trains and graphs in python.

#### Please look at the [documentation](https://github.com/Ziaeemehr/itng_toolbox/blob/master/itng/doc/build/latex/itngtoolbox.pdf).

### installation

#### from pip
```sh
pip3 install itng
```

#### from source code
```sh
sudo python3 setup.py install
```

-  to update the documentation:

```sh
cd itng/doc 
make html       # for html files
make latexpdf   # for pdf file
```
-  to run the test
```sh
cd itng/tests/
python3 test*.py
```

### How to contribute
#### Step 1: Set up a working copy on your computer

```sh
git clone git@github.com:Ziaeemehr/itng_toolbox.git
git remote add origin2 git@github.com:YOUR_USERNAME/itng_toolbox.git
git remote -v
```
#### Step 2: Do some work and push changes

```sh
git checkout -b new_branch_name
git add [files added or changed]
git commit -m "message to commit"
git push origin2 new_branch_name
```
then go to the repository and click on the green button **pull request**.



