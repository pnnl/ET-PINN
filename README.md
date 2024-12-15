# ET-PINN
This library offers tools for the efficient training of physics-informed machine learning models, using PyTorch as its backend and supporting both CPU and GPU training. It provides a flexible framework that allows users to configure their scientific problems, develop network models, manage network training components, and conduct grid tests. For advanced learning techniques, the library currently includes self-adaptive weighting and algorithms to enhance training performance. Later updates will integrate multi-fidelity learning approaches and self-adaptive sampling algorithms.


## Test examples
* Test cases in the paper "[Self-adaptive weights based on balanced residual decay rate for physics-informed neural networks and deep operator networks](https://arxiv.org/abs/2407.01613)"


## Run the examples
* Add the source code directory to your Python path.
* Copy the example test cases from the examples directory into your local working directory.
* In each subfolder, run the cases.py script to start the network training process.

We have also provided several batch files (run.sh, test.sh, and testaccuracy.sh) to simplify running the tests on a Linux cluster environment.

## LICENSE
This project is licensed under the GNU General Public License v3.0 (GPL v3.0). However, parts of this project include code from external libraries that are licensed under the GNU Lesser General Public License v2.1 (LGPL v2.1). These parts are located in the folder "ETPINN/geometry". For details about the LGPL v2.1 license, please refer to the COPYING.LESSER file in the folder "ETPINN/geometry".
 