# HiMICd-NCP: A daily 1 km atmospheric moisture index collection over the North China Plain during 2003–2020

##  1.Institutional information

### *1.1 Organization:* 

1. School of Geography and Planning, and Guangdong Key Laboratory for Urbanization and Geo-simulation, Sun Yat-sen University, Guangzhou 510006, China.
2. Key Laboratory of Watershed Geographic Sciences, Nanjing Institute of Geography and Limnology, Chinese Academy of Sciences, Nanjing 210008, China.
3. Jiangsu Provincial Key Laboratory for Advanced Remote Sensing and Geographic Information Science and Technology, Key Laboratory for Land Satellite Remote Sensing Applications of Ministry of Natural Resources, International Institute for Earth System Science, Nanjing University, Nanjing, Jiangsu 210023, China.


##  2.Contact information

### *2.1 Authors:*

Mrs. Zhiying Su<sup>1</sup> (suzhy9@mail2.sysu.edu.cn)

Dr. Hui Zhang^2^ (zhanghui@niglas.ac.cn)

Mr. Xiang Li^3^ (xiangli_nju@foxmail.com)

Dr. Sijia Wu^1^ (wusj8@mail.sysu.edu.cn)

Mrs. Manqing Shi^1^ (shimq7@mail2.sysu.edu.cn)

Dr. Ming Luo^1^ (luom38@mail.sysu.edu.cn)


##  3.Codes information

### *3.1 Input Data Preprocessing.py*

Gridded datasets processing, extraction values using weather station and data cleaning are recorded in this file. The section called “Data Acquisition” should be performed in Google Earth Engine, and the rest is operational under Python 3.10.

### *3.2 HiMICd-NCP Model.py*

The codes in “HiMICd-NCP Model.py” is used for model training and prediction, as well as for incorporating accuracy evaluation.

### *3.3 Result Visualization.py*

The overall accuracy assessment of the six indices and the accuracy visualization code are also publicly available in the file named “Result Visualization.py” and the sample data required is publicly available in the “Data Samples” folder.
