# AI-Graph
The project is to train artificial intelligent models to recognize abnormal traffic graphs or KPI graphs. It can be used by performance or IT engineers to figure out abnormal graphs from thousands of traffic graphs like cacti or KPI graphs from telecom OSS.

First release, ai_cacti_test.py with pre-trained model pm_graph_variables5.ckpt identifies three types of graph: 1) Sudden traffic drop graphs indicating outage; 2) Plateau graphs indicating capacity limitation; 3) Normal graphs which exclude outage and plateau graphs.<br />
![alt text](https://raw.githubusercontent.com/bryandu/AI-Graph/master/abnormal_graphs.png)
It uses five conventional neural network layers and one full connected layers as follow,
![alt text](https://raw.githubusercontent.com/bryandu/AI-Graph/master/pm_graph_model.png)


**Installation Procedure:**<br />
1. Install python3.

2. Install tensorflow 1.4. Tensorflow versions higher than 1.4 might have compatible issues.<br />

&nbsp;&nbsp;&nbsp;&nbsp;*virtualenv --system-site-packages ~/venvs/tensorflow*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*source ~/venvs/tensorflow/bin/activate*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*sudo pip3 install tensorflow==1.4*<br />

3. Install required python3 module.<br />

&nbsp;&nbsp;&nbsp;&nbsp;*sudo pip3 install pysftp<br />
&nbsp;&nbsp;&nbsp;&nbsp;sudo pip3 install numpy<br />
&nbsp;&nbsp;&nbsp;&nbsp;sudo pip3 install python36-tkinter*<br />

4. Download AI_Graph_master.zip from Github and unzip to local AI-Graph-master directory.<br />

5. Download the following two files(too big, cannot upload to github) and put into AI-Graph-master directory.<br />

&nbsp;&nbsp;&nbsp;&nbsp;pm_graph_variables5.ckpt.data-00000-of-00001<br />
https://drive.google.com/open?id=1nhWdIWnYf6ywCWPpN3OonRb_yNVgu0mT

&nbsp;&nbsp;&nbsp;&nbsp;gi-imperil.npy<br />

https://drive.google.com/open?id=1PKVtxquGk_FxKIKN1WdWboc8OKvIntGz

5. Install nmidDataExport v1.1.0. <br />
https://www.urban-software.com/products/nmid-plugins/nmiddataexport/

6. Replace setup.php files in the nmiddataexport directory. I change setup.php to export 24h graph data. <br />

7. In cacti server, select Console/Devices/Graph Lists/Graph management/Automated Export - Add to Export. Then Cacti will output 24h graph data to /usr/share/cacti/plugins/nmidDataExport/export/

**Run:**<br />
1. Test any cacti csv files in .\pm_graph\data\test<br />
&nbsp;&nbsp;&nbsp;&nbsp;*python3 ai_cacti_test.py*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CSV file format is shown as follow. Title and Step field are optional. It must have Date column to let ai_cacti_test.py to extract in/out data.<br />

&nbsp;&nbsp;&nbsp;&nbsp;*Title:	'Great Plains GPC/IMPRENEETH/1023 PRI'	*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*Step:	60*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*Total Rows:	1440*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*Graph ID:	2945*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*Host ID:	30*<br />
		
&nbsp;&nbsp;&nbsp;&nbsp;*Date &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In &nbsp; Out*<br />
&nbsp;&nbsp;&nbsp;&nbsp;*5/2/2018 9:31 &nbsp;&nbsp; 1.6 &nbsp; 4.83*<br />

2. Test nmidDataExport files in cacti server at /usr/share/cacti/plugins/nmidDataExport/export/. ai_cacti_test.py sftp the csv files in cacti server at /usr/share/cacti/plugins/nmidDataExport/export/ and checks which one is abnormal.<br />
--s specify the cacti server name.<br />
&nbsp;&nbsp;&nbsp;&nbsp;*python3 ai_cacti_test.py --s cacti*

3. You can put the new 24h training data in .\pm_graph\data\normal, .\pm_graph\data\outage, .\pm_graph\data\plateau and run the following command to tune the model.<br />
&nbsp;&nbsp;&nbsp;&nbsp;*python3 ai_cacti.py --d ./data/ --v pm_graph_variables5.ckpt*
