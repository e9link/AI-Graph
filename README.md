# AI-Graph
Train an artifical intelligent model to recognize abnormal traffic graphs or KPI graphs. Great for performance or IT engineers to figure out abnormal graphs from thousands of traffic graphs like cacti or KPI graphs from telecom OSS.

First release, ai-cacati, for traffic graphs using cacti will be available on March 18th.

Second release, ai-kpi, for mobile RAN and core KPIs will be available on May 18th.

Installation:
1. Install python3.

2. Install tensorflow 1.4
virtualenv --system-site-packages ~/venvs/tensorflow
source ~/venvs/tensorflow/bin/activate
sudo pip3 install tensorflow==1.4

3. Install required python3 module.
sudo pip3 install pysftp
sudo pip3 install matplotlib
sudo yum -y install python36-tkinter

4.Download pm_graph from Github.
5.Download pre-trained files and put into ai-graph directory.

pm_graph_variables5.ckpt.data-00000-of-00001
https://drive.google.com/open?id=10Y_cdwICXpHABmVevbmKDo83aekJfp9N

gi-imperil.npy
https://drive.google.com/open?id=1PKVtxquGk_FxKIKN1WdWboc8OKvIntGz

Run:
Test any cacti csv files in .\pm_graph\data\test
>python3 ai_cacti_analyzer.py --s test

Train classified in .\pm_graph\data\normal, .\pm_graph\data\outage, .\pm_graph\data\plateau
>python3 ai_cacti.py --t data
