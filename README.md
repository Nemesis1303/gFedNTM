# gFedNTM
A gRPC based federated learning framework for neural topic models.

## Generate gRPC Python files from proto file
Get into venv:
```
python3 -m venv venv
virtualenv --python=/opt/homebrew/bin/python3 venv        
```
```
source venv/bin/activate
```
Generate files:
```
python3 -m grpc_tools.protoc -I ../protos --python_out=. \
        --grpc_python_out=. ../protos/federated.proto
```

The generated file "federated_pb2.py" contains the type definitions, while "federated_pb2_grpc.py" describes the framework for a client and a server.

## Run server

```
python3 main.py --min_clients_federation <min_clients_federation> --model_type <underlying_model_type> 
```

## Run client 

```
python3 main.py --id <id> --source <path_to_training_data> 
```
