# setting cuda on ubuntu 18.4

#### install nvidia 418

```bash
dpkg -l | grep -i nvidia
sudo apt-get remove --purge '^nvidia-.*'
echo 'nouveau' | sudo tee -a /etc/modules
sudo rm /etc/X11/xorg.conf
sudo apt-get remove --purge '^nvidia-.*'

sudo apt autoremove
sudo apt autoclean

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-418 nvidia-settings
```

### install CUDA Toolkit 10.1 

[sources](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)


```bash
sudo chmod +x /home/deny/Downloads/cuda_10.1.168_418.67_linux.run
/home/deny/Downloads/cuda_10.1.168_418.67_linux.run --override
```

### install cuDNN Library for Linux 
[sources](https://developer.nvidia.com/rdp/cudnn-download)

```bash
cd /home/deny/Downloads
tar -zxvf cudnn-10.1-linux-x64-v7.6.1.34.tgz
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/
sudo cp  cuda/include/cudnn.h /usr/local/cuda-10.1/include/
# Give read access to all users
sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

### install libcupti

```bash
sudo apt-get install libcupti-dev
```

### set path env ~/.bashrc or ~/.zshrc

```bash
echo "export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
source ~/.bashrc
```
### install tensorflow gpu

```bash
pip install --upgrade tensorflow-gpu
```


