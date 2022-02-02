import threading

class Federation(object):
    def __init__(self):
        self.federation_lock = threading.RLock()
        self.federation = {}

    def connect(self, client):
        print("Client {} connecting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
            else:
                self.federation[client] += 1

    def disconnect(self, client):
        print("Client {} disconnecting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                raise RuntimeError("Tried to disconnect client '{}' but it was never connected.".format(client))
            self.federation[client] -= 1
            if self.federation[client] == 0:
                del self.federation[client]

    def getClients(self):
        with self.federation_lock:
            return self.federation.keys()