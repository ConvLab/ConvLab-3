import convlab

from convlab.dst.emodst.tracker import EMODST
from convlab.nlu.jointBERT.unified_datasets.nlu import BERTNLU
from convlab.dst.rule.multiwoz.dst import RuleDST
from argparse import ArgumentParser


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["setsumbt", "trippy", "rule"], help="mode of the tracker")
    parser.add_argument("--setsumbt", type=str,
                        help="path to the setsumbt model weight")
    parser.add_argument("--trippy", type=str,
                        help="path to the trippy model weight")
    parser.add_argument("--ertod", type=str,
                        help="path to the ertod model weight")
    return parser.parse_args()


def get_tracker(args):
    print(args.mode)

    if args.mode == 'setsumbt':
        dst_model_name = 'setsumbt'
        kwargs_for_dst = {
            # path to the setsumbt repository
            'model_name_or_path': args.setsumbt,
        }
    elif args.mode == 'trippy':
        dst_model_name = 'trippy'
        kwargs_for_dst = {
            # path to the trippy repository on huggingface
            'model_path': args.trippy,
        }
    elif args.mode == 'rule':
        dst_model_name = "bertnlu"
        kwargs_for_dst = {
            'mode': 'user',
            'config_file': 'multiwoz21_user.json',
            'model_file': "https://huggingface.co/ConvLab/bert-base-nlu/resolve/main/bertnlu_unified_multiwoz21_user_context0.zip"
        }

    tracker = EMODST(
        kwargs_for_erc={
            'base_model_type': 'bert-base-uncased',
            'model_type': 'contextbert-ertod',
            # path to the contextbert checkpoint
            'model_name_or_path': args.ertod,
        },
        dst_model_name=dst_model_name,
        kwargs_for_dst=kwargs_for_dst
    )
    return tracker


def main():
    args = arg_parser()
    tracker = get_tracker(args)

    tracker.init_session()

    # # prepending empty strings required by trippy
    # tracker.dst.state['history'].append(['usr', ''])
    # tracker.dst.state['history'].append(['sys', ''])
    user_act = 'hey. I need a cheap restaurant.'
    state = tracker.update(user_act)
    # print(state)
    print(user_act)
    print(tracker.get_emotion())

    tracker.dst.state['history'].append(
        ['usr', 'hey. I need a cheap restaurant.'])
    tracker.dst.state['history'].append(
        ['sys', 'There are many cheap places, which food do you like?'])
    user_act = 'If you have something Asian that would be great.'
    state = tracker.update(user_act)
    # print(state)
    print(user_act)

    print(tracker.get_emotion())

    tracker.dst.state['history'].append(
        ['usr', 'If you have something Asian that would be great.'])
    tracker.dst.state['history'].append(
        ['sys', 'The Golden Wok is a nice cheap chinese restaurant.'])
    tracker.dst.state['system_action'] = [['inform', 'restaurant', 'food', 'chinese'],
                                          ['inform', 'restaurant', 'name', 'the golden wok']]
    user_act = 'Fuck!'
    state = tracker.update(user_act)
    print(user_act)
    print(tracker.get_emotion())
    # print(state)

    # tracker.state['history'].append(['usr', 'Great. Where are they located?'])
    # state = tracker.state
    # state['terminated'] = False
    # state['booked'] = {}


if __name__ == "__main__":
    main()
