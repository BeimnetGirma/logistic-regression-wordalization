# A single individual (row) from the model data, with its feature values and contributions
class Individual:

    def __init__(self, id, ser_metrics):
        self.id = id
        self.ser_metrics = ser_metrics
