from configparser import ConfigParser

config = ConfigParser()


config["globals"] = {
    "loss": "cross_entropy",
    "lrate": 0.1,
    "wreg": 0.001,
    "wrt": "L2"
} 

config["layers"] = {
    "input": {"size": 20},
    "1": {"size": 1, "act": "relu", "wr": [-0.1, 0.1], "lrate": 0.01},
    "2": {"size": 1, "act": "relu", "wr": "glorot", "br": [0, 1]},
    "output" : {"type": "softmax"}
}

with open("./model_config.txt", "w") as f:
    config.write(f)
