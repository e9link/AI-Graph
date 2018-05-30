# AI-Graph
Train an artifical intelligent model to recognize abnormal traffic graphs or KPI graphs. Great for performance or IT engineers to figure out abnormal graphs from thousands of traffic graphs like cacti or KPI graphs from telecom OSS.

First release, ai_cacti_test.py with pre-trained model pm_graph_variables5.ckpt identifies three types of graph: 1) Sudden traffic drop graph indicating outage; 2) Plateau graph indicating bandwidth limitation; 3) Normal graphs which exclude outage and plateau graphs.

![alt text](https://raw.githubusercontent.com/bryandu/AI-Graph/master/abnormal_graphs.png)
        
It uses four conventional neural network layers and one full connected layers as follow,
![alt text](https://raw.githubusercontent.com/bryandu/AI-Graph/master/pm_graph_model.png)


Installation Procedure:
1. Install python3.

2. Install tensorflow 1.4. Tensorflow version higher than 1.4 might have compatible issues.

virtualenv --system-site-packages ~/venvs/tensorflow
source ~/venvs/tensorflow/bin/activate
sudo pip3 install tensorflow==1.4

3. Install required python3 module.

sudo pip3 install pysftp
sudo pip3 install numpy
sudo pip3 install python36-tkinter

4.Download AI_Graph_master.zip from Github and unzip to local AI-Graph-master directory.

5.Download the following two files(too big, cannot upload to github) and put into AI-Graph-master directory.

pm_graph_variables5.ckpt.data-00000-of-00001
https://drive.google.com/open?id=1nhWdIWnYf6ywCWPpN3OonRb_yNVgu0mT

gi-imperil.npy

https://drive.google.com/open?id=1PKVtxquGk_FxKIKN1WdWboc8OKvIntGz

5. Install nmidDataExport v1.1.0. 
https://www.urban-software.com/products/nmid-plugins/nmiddataexport/

6. Replace setup.php files in the nmiddataexport directory. I change setup.php to export 24h graph data. 

7. In cacti server, select Console/Devices/Graph Lists/Graph management/Automated Export - Add to Export.
Then Cacti will output 24h graph data to /usr/share/cacti/plugins/nmidDataExport/export/

Run:
Test any cacti csv files in .\pm_graph\data\test
>python3 ai_cacti_test.py

Test nmidDataExport files in cacti server at /usr/share/cacti/plugins/nmidDataExport/export/. 
--s specify the cacti server name.
>python3 ai_cacti_test.py --s cacti

You can put the new 24h training data in .\pm_graph\data\normal, .\pm_graph\data\outage, .\pm_graph\data\plateau and run the following command to tune the model.
>python3 ai_cacti.py --d ./data/ --v pm_graph_variables5.ckpt
