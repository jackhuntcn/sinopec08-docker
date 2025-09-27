# sinopec08-docker

`sudo vim /etc/docker/daemon.json` 添加镜像

```
{
  "registry-mirrors": [
    "https://docker.xuanyuan.me",
    "https://mirror.ccs.tencentyun.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://docker.nju.edu.cn",
    "https://dockerproxy.com",
    "https://docker.mirrors.sjtug.sjtu.edu.cn/",
    "https://mirror.baidubce.com"
  ]
}
```

重启 docker 服务

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

构建 docker 镜像

```
sudo docker build -t my-model:v2 .
```

复现测试

```
sudo docker run --rm -v '/home/zhengheng/competitions/sinopec/08/data:/input' -v './output:/output' my-model:v2 /input /output
```

打包文件

```
sudo docker save -o ../my-ai-model-v2.tar my-model:v2
```
