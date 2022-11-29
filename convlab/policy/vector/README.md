# Vectoriser

The vectoriser is a module used by the policy network and has several functionalities

1. it translates the semantic dialogue act into a vector representation usable for the policy network
2. it translates the policy network output back into a lexicalized semantic act
3. it creates an action masking that the policy can use to forbid illegal actions

There is a **vector_base** class that has many functionalities already implemented. All other vector classes are inherited from the base class.

If you build a new vectoriser, you need at least the following method:

    
    def state_vectorize(self, state):
        # translates the semantic dialogue state into vector representation
        # will be used by the policy module
    
See the implemented vector classes for examples.