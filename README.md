# Logistics Optimization based on Deep Reinforcement Learning

- School: University of Science, VNU-HCM
- Faculty: Information Technology
- Class: Honor 2017
- Advisors: 
  - Assoc. Prof. Ly Quoc Ngoc
  - MSc. Pham Minh Hoang

| Student | ID | Phone Number |
|--|--|--|
| Le Tuan Dat | 1712329 | 0941623569 | 
| Pham Cao Vi | 1712902 | 0388177514 |

## Dependencies
- [PyTorch](https://pytorch.org/)
- [gym-OpenAI](https://gym.openai.com/)
- [atari-py](https://pypi.org/project/atari-py/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/index.html)
- [Matplotlib](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)
- [PyGame](https://www.pygame.org/news)
- [RL-lib](https://github.com/ledat2110/RL_lib)

## Modules

* **lib**: contain the envivronment simulation code and network model code.
* **model**: contain the trained network model.
* **requirements.txt**: contain the library used.
* **vpg_method.py**: code train the agent of vanilla policy gradient method.
* **matrix_vpg_method.py**: code train the agent of vanilla policy gradient method with proposed model.
* **retailer_vpg_method.py**: code train the agent running the retailers by vanilla policy gradient method.
* **warehouse_vpg_method.py**: code train the agent running the plant, the warehouse by vanilla policy gradient method.
* **test.py**: test the trained agent.
* **web_demo.py**: run the demo in web.

## Setup environment
### Setup with sh file
`bash setup_lib.sh`
### Setup by command line
`pip install RL_lib`

`pip instlal -r requirements.py`

## TRAIN
### VPG agent
`python vpg_method.py -n NAME_RUNTIME [-m DIR_PRETRAINED_MODEL] [-s]`
#### Arguments
* `-n, --name`: name of the runtime. This is required.
* `-m, --model`: dir of the pretained model.
* `-s, --stop`: nostop traininng.

### Matrix VPG agent
`python matrix_vpg_method.py -n NAME_RUNTIME [-m DIR_PRETRAINED_MODEL] [-s]`
#### Arguments
* `-n, --name`: name of the runtime. This is required.
* `-m, --model`: dir of the pretained model.
* `-s, --stop`: nostop traininng.


### Multi VPG agent
#### Retailer agent
`python vpg_method.py -n NAME_RUNTIME [-m DIR_PRETRAINED_MODEL] [-s]`
#### Arguments
* `-n, --name`: name of the runtime. This is required.
* `-m, --model`: dir of the pretained model.
* `-s, --stop`: nostop traininng.

#### Warehouse agent
`python vpg_method.py -n NAME_RUNTIME -rm DIR_RETAILER_MODEL [-m DIR_PRETRAINED_MODEL] [-s]`
#### Arguments
* `-n, --name`: name of the runtime. This is required.
* `-rm, --retailer_model`: dir the of the pretrained retailer agent's model. This should be included.
* `-m, --model`: dir of the pretained model.
* `-s, --stop`: nostop traininng.

## TEST
`python test.py -t TYPE_OF_AGENT -m DIR_PRETRAINED_MODEL [-n NUMBER_OF_EPISODE] [-p] [-tr] [-v] [-rd] [-br] [-rm DIR_PRETRAINED_RETAILER_MODEL]`
### Arguments
* `-t, --type`: type of the agent with `[vpg, matrix_vpg, 2_agent]`.
* `-m, --model`: path to the pretrained model.
* `-n, --n_episode`: the number of running episode.
* `-p, --plot`: show the chart of values in logistics.
* `-tr, --trend`: activate trend demand.
* `-v, --var`: activate variance demand.
* `-rd, --random_demand`: activate random demand.
* `-br, --break_sp`: break event in running the logistics process.
* `-rm, --retailer_model`: the path to the pretrained retailer model if use `2_agent` type.

## DEMO
Run the demo on website

`streamlit run web_demo.py`
