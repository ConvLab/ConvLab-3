from convlab.policy.genTUS.stepGenTUS import UserPolicy
from convlab.dialog_agent import PipelineAgent
from convlab.policy.tus.unify.util import create_goal
from convlab.policy.genTUS.unify.build_data import DataBuilder
from convlab.util import load_dataset


def main():
    # initialize user policy, here we take a default user policy trained on MultiWOZ
    usr_policy = UserPolicy(mode="semantic")
    usr_nlu = None
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')

    # get a goal from sgd dataset
    data_builder = DataBuilder("sgd")
    data = load_dataset("sgd")
    g = data_builder.norm_domain_goal(create_goal(data["train"][0]))

    # initial the user with the goal from dataset
    usr.init_session(goal=g)

    # a mock conversation
    print(usr.response([]))
    print(usr.response([['request', 'Restaurants', 'area', '?']]))
    print(usr.response([['request', 'Restaurants', 'city', '?']]))


if __name__ == "__main__":
    main()
