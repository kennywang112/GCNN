
Data: [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) <br/>
GCN framework: [GCNconv](https://arxiv.org/pdf/1609.02907) <br/>
Ref : [MRE-CNN](https://arxiv.org/pdf/1807.10575) <br/>
Mediapipe model: [face_landmarker](https://huggingface.co/lithiumice/models_hub/blob/8a7b241f38e33d194a06f881a1211b3e7eda4edd/face_landmarker.task)
## How to run
1. Install data from **kaggle** and put it in **Image_data** folder 
2. Install **face_landmarker** and put it in **model** folder
3. Run RunAdjacency.py to get image matrix
4. Run RunModel.py for the model and Grad-Cam

### Finally you get a full folder structure below
```
path/GCNN
├─.ipynb_checkpoints
├─GradCam
├─Image_data
│  └─DATASET
│      ├─test
│      │  ├─1
│      │  ├─2
│      │  ├─3
│      │  ├─4
│      │  ├─5
│      │  ├─6
│      │  └─7
│      └─train
│          ├─1
│          ├─2
│          ├─3
│          ├─4
│          ├─5
│          ├─6
│          └─7
├─model
├─output_data
│  ├─adjacency
│  │  ├─adjacency_1
│  │  ├─adjacency_2
│  │  ├─adjacency_3
│  │  ├─adjacency_4
│  │  ├─adjacency_5
│  │  ├─adjacency_6
│  │  └─adjacency_7
│  └─landmarks
```
## Note
Install torch with the command below to get a GPU version (if any)<br/>
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/metal.html` <br/>


## Contributors & Collaborators

<table>
  <tbody>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hihisw"><img src="https://avatars.githubusercontent.com/u/91866927?v=4" width="100px;" alt="尹士文"/><br /><sub><b>尹士文</b></sub></a>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cys0107"><img src="https://avatars.githubusercontent.com/u/91866945?v=4" width="100px;" alt="周宇舒"/><br /><sub><b>周雨舒</b></sub></a>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hsin6"><img src="https://avatars.githubusercontent.com/u/91867022?v=4" width="100px;" alt="張藝馨"/><br /><sub><b>張藝馨</b></sub></a>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bunnnn1002"><img src="https://avatars.githubusercontent.com/u/91866935?v=4" width="100px;" alt="蕭邦宇"/><br /><sub><b>蕭邦宇</b></sub></a>
  </tbody>
</table>