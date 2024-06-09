# General workflows

Here we will go over the steps to do some of the main tasks, from training a YOLO model on custom data to running simulations with different
configurations. All of the main Workflows have a dedicated, interactive notebook (.ipynb file) ready to use with explanations for each step. All
of the workflow notebooks are located in a dedicated folder called "workflows".

## Workflow Files Descriptions

`create_yolo_images.ipynb` - Prepares raw frames of some experiment for the process of training YOLO model on them. This step entails
detecting the worm in selected frames and cropping a region of pre-defined size around the worms.

`yolo_training.ipynb` - Used to train a YOLO model on a given dataset. The training dataset was prepared by annotating 3 the images which
were extracted using the notebook create_yolo_images. The annotation process can be done with RoboFlow, which is an online dataset
creation and annotation tool.

`initialize_experiment.ipynb` - In order to run system simulations on a new experiment, first it’s essential to initialize the experiment. The
initialization step runs the YOLO predictor on the raw experiment, detects worm’s head position in each frame and saves the detection results
into a log. That log would be later used for simulating different control algorithms on the experiment. In addition, the background image and
worm images are extracted from the raw frames. These can be used later during analysis, to calculate the segmentation based error. This log is
useful since in the future the simulator can simply read worm head positions from the log, instead of using YOLO to predict worm’s head
position in every frame of interest (which is much slower, especially on computers without a dedicated graphics card).

`simulate.ipynb` - Run a full system simulation on some previously initialized experiment. The simulation is ran by reading an experiment
log produced by the initialization process - in each frame, worm’s head position is retrieved from the log. In this notebook it is possible to
simulate the system with any controller and any configuration parameters, not only the ones of used for the initial experiment log. Similar to
the initialization process, the simulation produces a log, which would be later used to analyze system’s performance and its behavior.

`analysis.ipynb` - This notebook is used to analyze the performance of a control algorithm (controller). A log which was produced by running
simulate is read and analyzed, and different plots and statistics are presented. In addition, there is an option to calculate segmentation
evaluation-error, by counting how many pixels of the worm are outside of the microscope view. To this end, we use the background and worm
images which were extracted during the run of intialize_experiment notebook for this experiment.

`visualize.ipynb` - Given a system log which was produced by simulate, this notebook is able to visually recreate the simulator’s behavior. At
each frame, the position of worm’s head is drawn, the position of the microscope FOV, and also the camera FOV. This notebook is used to
visually assess the performance and the behavior of the simulator, and to visually investigate what causes the system to misbehave.
predictor_training - Used to train a specific simulation control algorithm. The MLPController is an algorithm that uses a neural
network (NN) to predict worm’s future position. Since this algorithm is NN based, it requires training. That script is responsible to train that
NN from experiment log files, which were produced by either running initialize or simulate (doesn’t matter).

`polyfit_optimizer.ipynb` - This notebook is used to tune the parameters of a specific simulation control algorithm. The PolyfitController
is an algorithm that uses polynomial-fitting to predict worm’s future position. A polynomial is fitted from past observations at previous time
stamps, and afterwards sampled in the future time to predict worm’s position. This notebook is used to determine the optimal degree of the
fitted polynomial, and to find the optimal weight of each past sample for the fitting process.