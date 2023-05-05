# Brief Description:
The following files are included:
- `q1.py`: The python code for Q1
- `q2.py`: The python code for Q2
- `Q1_output.txt`: The file containing the output contents for q1.py and the tree data in a hierarchical order (Split indicates the best attribute on the basis of which the split was made, Best Value indicates the best value that can be achieved if further child nodes are made inaccessible)
- `Q1_plot.png`: The file containing the plot of accuracy vs depth obtained after pruning the tree in q1.py
- `Q2_output.txt`: The file containing the output contents for q2.py

# Directions to use the code  
1. Download this directory into your local machine

2. Copy the  `Dataset_C.csv` file in the Source Code Directory

3. Ensure all the necessary dependencies with required version and latest version of Python3 are available <br>
 `pip3 install -Iv numpy==1.23.2 matplotlib==3.5.3 pandas==1.4.3`

# For running the python code for Question 1
- Run <br>
`python3 q1.py`
- The output would be saved to `Q1_output.txt` and the relevant post-pruning plot of Accuracy vs Depth `Q1_plot.png` would be generated

# For running the python code for Question 2
- Run <br>
`python3 q2.py`
- The output would be saved to `Q2_output.txt`