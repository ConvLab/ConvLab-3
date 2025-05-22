from convlab.base_models.t5.nlu import T5NLU
from convlab.dialog_agent import PipelineAgent
from convlab.policy.genTUS.stepGenTUS import UserPolicy
from convlab.policy.genTUS.unify.build_data import DataBuilder
from convlab.policy.tus.unify.util import create_goal
from convlab.util import load_dataset


def main():
    # initialize user policy, here we take a default user policy trained on MultiWOZ
    usr_policy = UserPolicy(
        model_checkpoint="convlab/policy/genTUS/unify/experiments/sgd/22-10-10-14-15",
        dataset="sgd",
        mode="language")
    usr_nlu = None
    usr_nlu = T5NLU(speaker='system', context_window_size=0,
                    model_name_or_path='ConvLab/t5-small-nlu-multiwoz21')
    usr = PipelineAgent(usr_nlu, None, usr_policy, None, name='user')

    # get a goal from sgd dataset
    data_builder = DataBuilder("sgd")
    data = load_dataset("sgd")
    g = data_builder.norm_domain_goal(create_goal(data["train"][1]))

    # initial the user with the goal from dataset
    print(g)
    print('-'*10)
    usr.init_session(goal=g)

    # a mock conversation
    print(usr.policy.get_goal())
    print(usr.response("Hi, welcome to the system. How can i help you"))
    print(usr.response("Do you have any specific location in mind?"))
    print(usr.response("Which price range?"))


if __name__ == "__main__":
    main()
