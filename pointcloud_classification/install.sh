#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install

# EMD
# cd $HOME/extensions/emd
# python setup.py install