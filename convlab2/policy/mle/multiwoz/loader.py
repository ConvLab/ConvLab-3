import os
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.mle.loader import ActMLEPolicyDataLoader


class ActMLEPolicyDataLoaderMultiWoz(ActMLEPolicyDataLoader):

    def __init__(self, vectoriser=None):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        if vectoriser:
            self.vector = vectoriser
        else:
            print("We use vanilla Vectoriser")
            self.vector = MultiWozVector()

        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')

        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir)

