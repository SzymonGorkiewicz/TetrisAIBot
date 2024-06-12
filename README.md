# Tetris reinforcement learning

Previous tetris project [tetris-python](https://github.com/SzymonGorkiewicz/tetris-python) with neural network implementation for reinforcement learning.

* Current neural network consist of **5 hidden layers** and **128 nodes**</br>
  ![neural](https://github.com/SzymonGorkiewicz/TetrisAIBot/assets/92310752/ff01004f-d763-4452-bc3a-1b28da2dbb81)
* With the forward function using reLu activation</br>
  ![forward](https://github.com/SzymonGorkiewicz/TetrisAIBot/assets/92310752/512a9ce6-b01d-4235-9a0f-8ddf0a7a801b)


## Setting up the project

1. Pull the repository.

2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements for the project.

```bash
pip install pygame
```

3. Create a virtual environment
```bash
python -m venv <name_of_your_environment>
```
4. Activate the virtual environment in the ***CMD***
```bash
<name_of_your_environment>\Scripts\activate
```
5. Install requirements in your created virutal environment
```bash
pip install -r requirements.txt
```
6. Now open pulled repository through code editor ( [Visual Studio Code](https://code.visualstudio.com) preffered ).
   * select your virtual environment as a code interpreter 
   ![interpreter](https://github.com/SzymonGorkiewicz/TetrisAIBot/assets/92310752/14059b09-8ee3-4866-8be8-8e5f882cd63e)
   * path to your interpreter
     ```bash
     <name_of_your_environment>\Scripts\python.exe
      ```
   * if you add interpreter correctly you should see it in the right corner of the VS code editor.
     ![venv](https://github.com/SzymonGorkiewicz/TetrisAIBot/assets/92310752/aa18569b-0d2a-48f6-8465-e5c1c5cabb02)

7. If you've done everything correct you can run project through ***agent.py*** file.
   * You should see this:</br>
     ![game](https://github.com/SzymonGorkiewicz/TetrisAIBot/assets/92310752/6e197dcf-e1d6-4bb7-b848-9478d3ec33b4)
# Notice that the project is still in learning mode. Currently loaded model is after 12_000 games. He is doing pretty good job on eliminating holes, but unfortunately there is still something more to do in implementating reward function.



## Contributing

Pull requests are welcome.
