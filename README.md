# MuJoCo OpenAI Gym Ubuntu
Short description of installation process for MuJoCo-Py on MuJoCo 1.50 with OpenAI Gym on Ubuntu 16.04  
**Note**: This is a compilation of my experiences and useful resources I found. Hope this can be a helpful little *how to guide*! 

Author: Gabriel Fernandez  
GitHub: https://github.com/gabriel80808 

## Ubuntu
Make sure to have [Ubuntu 16.04](https://www.ubuntu.com/download/desktop) installed. You can try with another version, but this is the one I am using.

## Anaconda
1. Download the latest version of [Anaconda](https://www.anaconda.com/download/#linux) with Python 3.6 version. My version of Anaconda is 4.5.1. You can probably use any other type of virtual environment. This allows us to isolate the environment we are setting up just incase it breaks.  
2. Create a virtual envinronment with Anaconda:  
```bash
foo@bar:~$ conda create -n NameOfYourEnvironment python=3.6 anaconda
```
3. Start your virtual environment:
```bash
foo@bar:~$ conda activate EnvName
```
**Note**: You can stay within your virtual environment throughout this process. Where you see `(MyEnv)` in this tutorial it is generally okay to be within your virtual environment. When you modify your python environment you want to be inside your virtual environment. Otherwise it's optional. If you're not sure, it's generally fine throughout this tutorial.

## MuJoCo
1. Go to [Mujoco](https://www.roboti.us/index.html) and follow directions to obtain a license. They have free licenses for students and a 30-free trial version.
2. Download MuJoCo 1.50 binaries for Linux. Right click on the file and go to properties to make sure that the file is executable before running it.
3. Unzip `mjpro1.50` to `~/.mujoco/mjpro150`
4. Place your license key, `mjkey.txt`, from step 2 to ~/.mujoco/mjkey.txt  
**Note**: This setup is primarily for [MuJoCo-Py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)

## Docker
1. See the [Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile) for the list of canonical dependencies. If you already know how to run docker files you can go to the next section.
2. See this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for a practical instillation guide of docker. Keep in mind that you will have to run `sudo` evertime you run docker unless you change your settings which is also explained in the link.

## MuJoCo-Py
1. Git clone the MuJoCo-Py repository in your hidden `.mujoco` directory:
```bash
(MyEnv)foo@bar:~/.mujoco$ git clone https://github.com/openai/mujoco-py.git
```
2. Copy your MuJoCo license key, `mjkey.txt`, into your recently downloaded repository directory:
```bash
(MyEnv)foo@bar:~/.mujoco$ cp mjkey.txt mujoco-py
```
3. Change your directory to the repository directory.
4. Now you can build your docker file:
```bash
(MyEnv)foo@bar:~/.mujoco/mujoco-py$ sudo docker build -t mujoco_doc .
```
**Note**: `-t` is just tagging the build with a name. Also there is a period at the very end to indicate the directory in which it will build. Take a look at this [simple tutorial](https://deis.com/blog/2015/creating-sharing-first-docker-image/) here.

## OpenAI Gym
1. Go to [OpenAI](https://github.com/openai/gym)'s GitHub  
**Note**: There are several ways to install the necessary packages needed. Keep in mind if you do a minimal instillation, you need to additionally download and install another package: `pip install -e '.[robotics]`. I will run through the full installation.
2. Make sure all of the necessary dependencies are there or installed:
```bash
(MyEnv)foo@bar:~/.mujoco/mujoco-py$ sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
3. Now we can install the full package:
```bash
(MyEnv)foo@bar:~/.mujoco/mujoco-py$ pip install 'gym[all]'
```

## Sanity Checks
**Note**: If you are not too familiar with how to debug some of the errors, read a little bit of the next section and come back here. It will be an iterative process. Good luck!
1. Start your virtual environment if you haven't done so:
```bash
foo@bar:~/.mujoco/mujoco-py$ conda activate YourEnvironmentName
```
2. Let's start off easy and pull up python and import mujoco_py:
```bash
(MyEnv)foo@bar:~/.mujoco/mujoco-py$ python

```
```python
>>> import mujoco_py
>>>
```
**Note**: If you see any errors or warnings, that means we got a lot of debugging to do. See the next section.  

3. Run a slightly modified example from MuJoCo-Py:  
**Note**: You should get the same results here unlike the next one. Remember when you load the model you will have to change `foo` to your directory name.
```python
>>> import mujoco_py
>>> model = mujoco_py.load_model_from_path("/home/foo/.mujoco/mujoco-py/xmls/claw.xml")
>>> sim = mujoco_py.MjSim(model)
>>> print(sim.data.qpos)
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
>>> sim.step()
>>> print(sim.data.qpos)
[  2.09217903e-06  -1.82329050e-12  -1.16711384e-07  -4.69613872e-11
  -1.43931860e-05   4.73350204e-10  -3.23749942e-05  -1.19854057e-13
  -2.39251380e-08  -4.46750545e-07   1.78771599e-09  -1.04232280e-08]
```
4. Now we can run a more involved check:
```python
>>> import mujoco_py
>>> import gym
>>> import numpy as np
>>> env = gym.make('HalfCheetah-v2')
WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype.
WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>. Please provide explicit dtype.
```
**Note**: Don't worry about the warning. It's a bug as far as I can tell. There's no way to fix the error without going into the code and basically removing it or initializing it in a function. Additionally, for this example you will get different values from me, so there's no need to freak out. Let's continue.
```python
>>> print(env.action_space)
Box(6,)
>>> print(env.observation_space)
Box(17,)
>>> init_observation = env.reset()
>>> init_observation
array([-0.09592622, -0.00758505,  0.04018139, -0.04553654, -0.03117339,
       -0.0567909 , -0.05794625,  0.01240102,  0.00091251,  0.10259654,
        0.03697612, -0.09892215, -0.00347502,  0.06516283,  0.03751707,
        0.15983774, -0.00057076])
>>> action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype='float32')
>>> env.render()
Creating window glfw
>>> obs, reward, done, info = env.step(action)
>>> obs
array([-0.09596615,  0.00432398,  0.01885579,  0.05236691, -0.02897269,
       -0.00611233, -0.00573494,  0.01972231,  0.27388623, -0.15326082,
        0.24379943, -0.05508176,  1.60848039,  1.13358126,  1.53246801,
        1.44243414,  0.29057963])
>>> reward
0.17909146134763032
>>> done
False
>>> env.render()
```

## Misc. Debugging Trickery
**Note**: You're almost there! This section will cover how to deal with a few errors I've come across: missing modules, GLFW3, GLEW, libOpenGL.so.0 & other lib errors associated with GPU CUDAs.

### Missing Modules  
As far as python modules go, you will usually get errors that say some module is missing. Simply `pip install` the missing module. If you receive longer compiling errors, especially those with the term `lib` in it, that usually means the GPU libraries are set up incorrectly. See the following sections.

### Modifying `.bashrc` for GLEW, GLFW3 Errors, and More
**Note**: `.bashrc` is a shell script that Bash runs when it starts. That means if you modify it, you have to open a new terminal to run the updated shell script. You can just append these lines to the bottom of your `.bashrc` file. It's a hidden file, so to see it you will need to type in `ls -al` into your terminal. You will need to also open `.bashrc` with an editor such as Sublime or you probably can just type `gedit`. You can probably do the following to get to the file:
```bash
(MyEnv)foo@bar:~$ gedit .bashrc
```
1. Add this line in `.bashrc` to fix MuJoCo-Py GLFW3 error:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
```
2. Add this line in `.bashrc` to fix MuJoCo-Py GLEW error:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-390/libGL.so
```
**Note**: The original thread on this error is [here](https://github.com/openai/mujoco-py/pull/145). Check what Nvidia driver you have installed on your system by following the directory path. Mine is `nvidia-390`. Change your number to match your driver. This also applies to the next step, 3.

3. Add this line in `.bashrc` for the MuJoCo-Py environment variable:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-390
```
### GPU CUDA Related Errors: libOpenGL.so.0, lib*, OpenGL*, Etc
**Note**: This does not go into the setup for CUDAs in Tensorflow. See Tensorflow's [guide](https://www.tensorflow.org/install/install_linux#NVIDIARequirements) and this useful [tutorial](http://www.python36.com/install-tensorflow141-gpu/) for that. I found that majority of the difficult errors arise from having an outdated driver or somehow the wrong one. In particular it was an Nvidia card for me. Later on I will mention how on an older version of MuJoCo-Py I avoided this by simply running it off of my Intel Core i7. I possibly may do a CUDA setup installation guide in the future. Cross my fingers.    
1. Find and keep note of GPU and other system information: 
```bash
foo@bar:~$ lspci | grep -i vga
00:02.0 VGA compatible controller: Intel Corporation 3rd Gen Core processor Graphics Controller (rev 09)
01:00.0 VGA compatible controller: NVIDIA Corporation GK107M [GeForce GT 640M] (rev a1)
foo@bar:~$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
```
**Note**: `lscpu` will produce a much longer list of information but we just want the first two lines. From the first command we typed, we found out that: My product type is GeForce, and my product series is GeForce 600M series. We are on Linux 16.04 if you forgot. The second command now tells us that my system can run 32-bit and 64-bit, and the architechture is set up for 64-bit. In the next step, 2, I want to fill in Linux 64-bit. If your system runs both, you probably want 64-bit for better performance.

2. Go to the [Nvidia Driver Search Page](https://linuxconfig.org/how-to-install-the-latest-nvidia-drivers-on-ubuntu-16-04-xenial-xerus) and fill out the fields with the information we collected from step 1. The version number for mine reads 390.48. Don't worry about what comes after the decimal. You can build from source, but I decided to just use `apt-get`. For now just take note of the driver number you need and don't worry what comes after the decimal place.

3. Assuming you already have a Nvidia driver installed. Restart your computer. On the start up GRUB boot menu make sure to highlight the Ubuntu entry and press the `E` key. It should bring you to a page you can edit. Don't delete anything. Just at the very end add `nouveau.modeset=0` and then press the `F10` key to continue booting.

4. On the login screen enter the keys `Ctrl` + `Alt` + `F1`. Then proceed to login with your normal credentials as if you were at your normal login screen.

5. Now we will delete everything related to Nvidia to start from fresh:
```bash
foo@bar:~$ sudo apt-get purge nvidia*
foo@bar:~$ sudo reboot
```

6. On the reboot repeat steps 3 and 4.

7. Once we are logged back in, we can install the Nvidia drivers from the proprietary GPU drivers' PPA:
```bash
foo@bar:~$ sudo add-apt-repository ppa:graphics-drivers/ppa
foo@bar:~$ sudo apt-get update
foo@bar:~$ sudo apt-get install nvidia-390 nvidia-prime
foo@bar:~$ sudo reboot
```
**Note**: Please remember to change the nvidia driver to the correct number in step 2. I additionally installed `nvidia-prime`. `nvidia-prime` allows the user to switch from the Nvidia card to Intel's. This came in handy in an older version of MuJoCo-Py. If things are really buggy, and you don't need the optimization using Intel's card might save a lot of headaches. To switch cards simply go to the Ubuntu Unity Dash (the search function) and type in Nvidia to pull up NVIDIA X Server Settings. Go to PRIME Profiles and you should see a little option to switch.

## Finished!
Hopefully, this was somewhat helpful. I am very new to the community. This is one of the ways I am trying to give back. Please leave comments or suggestions so that I can modify this to be better. I know how stressful it is getting this set up.
