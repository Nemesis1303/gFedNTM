# fuzzy-spoon
GRPC based federated learning framework for ProdLDA and Contextualized Topic Models

## Generate GRPC Python files from proto file

```
python3 -m grpc_tools.protoc -I ../protos --python_out=. \
        --grpc_python_out=. ../protos/federated.proto
```

The generated file "federated_pb2.py" contains the type definitions, while "federated_pb2_grpc.py" describes the framework for a client and a server.

## Run server

```
python3 server.py
```

## Run client 

```
python3 client.py -i <id> -p <period>
```
or

```
python3 client.py --id <id> --period <period>
```