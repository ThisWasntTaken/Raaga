# Raaga
1. Install CUDA from the NVIDIA installer site : https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
   a. Don't get alarmed by the screen flashing - happens when display drivers re-install
   b. Choose to install the libs/code/samples in appropriate locations. Trust me - you will thank me later!
2. Install Anaconda from their site : https://www.anaconda.com/products/individual#Downloads
3. Get pytorch from their site
   a. Will redirect you to run a command line according to the cuda version you installed. Most cases, you can choose a slightly older version here and not have any problems, as CUDA is nicely backward compatible
   b. Open an anaconda prompt and run that command line - and go grab a coffee... Takes a long while!
4. Install ffmpeg
   a. Download from here : https://www.gyan.dev/ffmpeg/builds/
      Very screwed up site - doesn't give links where they should be. Look at the bottom and avoid reading irrelevant stuff there
   b. Add to PATH : This is somehow so important... No python module understands the location - even if we explicitly initialize this.

