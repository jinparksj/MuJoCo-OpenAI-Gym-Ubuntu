# Mujoco-OpenAi-Gym-Ubuntu
Short description of instillation process for Mujoco 1.50 with OpenAi Gym

## Ubuntu
Make sure to have [Ubuntu 16.04](https://www.ubuntu.com/download/desktop) installed. You can try with another version, but this is the one I am using.

## Anaconda
Download the latest version of [Anaconda](https://www.anaconda.com/download/#linux) with Python 3.6 version. My version of Anaconda is 4.5.1. You can probably use any other type of virtual environment.

## Create Virtual Envinronment
In the command line:  
```bash
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
2. See this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for a practical instillation guide of docker. Keep in mind that you will have to run `sudo` evertime you run docker unless you change your settings which is also explained in the link.

## MuJoCo-Py
1. Git clone the MuJoCo-Py repository in your hidden `.mujoco` directory:
```bash
foo@bar:~/.mujoco$ git clone https://github.com/openai/mujoco-py.git
```
2. Copy your MuJoCo license key, `mjkey.txt`, into your recently downloaded repository directory:
```bash
foo@bar:~/.mujoco$ cp mjkey.txt mujoco-py
```
3. Change your directory to the repository directory.
4. Now you can build your docker file:
```bash
foo@bar:~/.mujoco/mujoco-py$ sudo docker build -t mujoco_doc .
```
**Note**: `-t` is just tagging the build with a name. Also there is a period at the very end to indicate the directory in which it will build. Take a look at this [simple tutorial](https://deis.com/blog/2015/creating-sharing-first-docker-image/) here.




### To Be Continued...
Ran into some issues with the recently released version that supports python 3.6






