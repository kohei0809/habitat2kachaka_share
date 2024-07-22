## TODO
・マップの端に移動した際のエラーを修正する<br>
(恐らくマップの周りにマージンを加えれば解決)<br>
・移動が失敗した次のステップの移動座標のバグを修正


## Install
```
cd habitat2kachaka_share/kachaka-api/python/demos
pip install -r requirements.txt
python -m grpc_tools.protoc -I../../protos --python_out=. --pyi_out=. --grpc_python_out=. ../../protos/kachaka-api.proto

cd habitat2kachaka_shape
pip install -r requirements.txt
```


## Usage
・habitat-simで学習したモデルの使用を想定<br>
・ppo_trainer.pyやenvironment.py、nav.py、ppo.pyなどを自身のプログラムから一部コピペしての使用を推奨
