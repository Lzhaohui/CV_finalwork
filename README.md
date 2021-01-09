# Sound-of-Pixels
ECCV18 "The Sound of Pixels".http://sound-of-pixels.csail.mit.edu/


## 环境安装
Windows10

python == 3.6.11
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy==1.2.1
pip install mir_eval
pip install opencv-python


## 训练
1.从MUSIC数据集上下载数据：https://github.com/roudimit/MUSIC_dataset

2.按照作者文中描述，将视频处理成8fps格式，音频处理成11025Hz采样率。

3.训练指令：
python -W ignore main.py --id MUSIC --list_train data/train.csv --list_val data/val.csv --arch_sound unet7 --arch_synthesizer linear --arch_frame resnet18dilated --img_pool maxpool --num_channels 32 --binary_mask 1 --loss bce --weighted_loss 1 --num_mix 2 --log_freq 1 --num_frames 3 --stride_frames 24 --frameRate 8 --audLen 65535 --audRate 44100 --num_gpus 1 --workers 48 --batch_size_per_gpu 20 --lr_frame 1e-4 --lr_sound 1e-3 --lr_synthesizer 1e-3 --num_epoch 100 --lr_steps 40 80 --disp_iter 20 --num_vis 40 --num_val 256

4.验证指令：
python -W ignore main.py --mode eval --id MUSIC-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride24-maxpool-binary-weightedLoss-channels32-epoch100-step40_80 --list_val data/val.csv --arch_sound unet7 --arch_synthesizer linear --arch_frame resnet18dilated --img_pool maxpool --num_channels 32 --binary_mask 1 --loss bce --weighted_loss 1 --num_mix 2 --log_freq 1 --num_frames 3 --stride_frames 24 --frameRate 8 --audLen 65535 --audRate 11025


## Reference
```bibtex
    @InProceedings{Zhao_2018_ECCV,
        author = {Zhao, Hang and Gan, Chuang and Rouditchenko, Andrew and Vondrick, Carl and McDermott, Josh and Torralba, Antonio},
        title = {The Sound of Pixels},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {September},
        year = {2018}
    }
```
