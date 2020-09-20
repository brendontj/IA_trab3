import pickle

class BaseAgent():

    def __init__(self, env):
        self.env = env
        self.snapshots = {}

    def save_snapshot(self, name):
        raise Exception('Method "save_snaptshot" not implemented')

    def save_snapshots_to_file(self):
        pickle.dump(self.snapshots, open('snapshot_lunarland.pickle', 'wb'))

    def load_snapshots_from_file(self):
        self.snapshots = pickle.load(open('snapshot_lunarland.pickle', 'rb'))

