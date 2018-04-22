# Mujoco-OpenAi-Gym-Ubuntu
Short description of instillation process for Mujoco 1.50 with OpenAi Gym

## Ubuntu
Make sure to have [Ubuntu 16.04](https://www.ubuntu.com/download/desktop) installed.

## Anaconda
Download the lateste version of [Anaconda](https://www.anaconda.com/download/#linux) with Python 3.6 version.  
Note that my version of Anaconda is 4.5.1

## Create Virtual Envinronment
In the command line:  
```
foo@bar:~$ conda create -n NameOfYourEnvironment python=3.6 anaconda
```
This allows us to isolate the environment we are setting up just incase it breaks.  
Now you have a virtual environment.

## MuJoCo
1. Go to [Mujoco](https://www.roboti.us/index.html) and follow directions to obtain a license. They have free licenses for students and a 30-free trial version.
2. Download MuJoCo 1.50 binaries for Linux. Right click on the file and go to properties to make sure that the file is executable before running it.
3. Unzip `mjpro1.50` to `~/.mujoco/mjpro150`
4. Place your license key, `mjkey.txt`, from step 2 to ~/.mujoco/mjkey.txt  
**Note**: This setup is primarily for [MuJoCo-Py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)

## Docker
1. See the [Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile) for the list of canonical dependencies. If you already know how to run docker files you can go to the next section.
2. See this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for a practical instillation guide of docker


### To Be Continued...
Ran into some issues with the new version that supports python 3.6


